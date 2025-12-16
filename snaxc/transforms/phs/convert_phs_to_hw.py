from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, comb, hw
from xdsl.ir import Block, SSAValue, TypeAttribute
from xdsl.parser import ArrayAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.mlir_opt import MLIROptPass

from snaxc.dialects import phs
from snaxc.phs.hw_conversion import get_choice_bitwidth, get_switch_bitwidth


class ConvertMuxes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, mux: phs.MuxOp, rewriter: PatternRewriter):
        # 0 = lhs, 1 = rhs
        # Change type of mux switch
        rewriter.replace_value_with_new_type(mux.switch, builtin.IntegerType(1))
        new_mux = comb.MuxOp(mux.switch, mux.rhs, mux.lhs)
        rewriter.replace_op(mux, [new_mux])


class ConvertPeOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, pe: phs.PEOp, rewriter: PatternRewriter):
        ports: list[hw.ModulePort] = []
        for i, data_opnd in enumerate(pe.data_operands()):
            ports.append(
                hw.ModulePort(
                    builtin.StringAttr(f"data_{i}"),
                    cast(TypeAttribute, data_opnd.type),
                    hw.DirectionAttr(data=hw.Direction.INPUT),
                )
            )
        for i, switch in enumerate(pe.get_switches()):
            ports.append(
                hw.ModulePort(
                    builtin.StringAttr(f"switch_{i}"),
                    builtin.IntegerType(get_switch_bitwidth(switch)),
                    hw.DirectionAttr(data=hw.Direction.INPUT),
                )
            )
        for i, output in enumerate(pe.get_terminator().operands):
            ports.append(
                hw.ModulePort(
                    builtin.StringAttr(f"out_{i}"),
                    cast(TypeAttribute, output.type),
                    hw.DirectionAttr(data=hw.Direction.OUTPUT),
                )
            )
        ports_attr = ArrayAttr(ports)
        mod_type = hw.ModuleType(ports_attr)
        new_op = hw.HWModuleOp(sym_name=pe.name_prop, module_type=mod_type, body=pe.body.clone())
        rewriter.replace_op(pe, new_op)


class ConvertChooseOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, choose_op: phs.ChooseOp, rewriter: PatternRewriter):
        yield_results: list[SSAValue] = []
        for region in choose_op.regions:
            for op in region.ops:
                # Move all non-yield operations outside the choice block
                if not isinstance(op, phs.YieldOp):
                    op.detach()
                    rewriter.insert_op(op)
                # put all yielded results in one big array
                else:
                    if not len(list(op.operands)) == 1:
                        raise NotImplementedError()
                    for operand in op.operands:
                        assert not isinstance(operand, Block)
                        yield_results.append(operand)
        rewriter.insert_op(create_array := hw.ArrayCreateOp(*yield_results))
        index_bw = get_choice_bitwidth(choose_op)
        rewriter.replace_value_with_new_type(choose_op.switch, builtin.IntegerType(index_bw))
        rewriter.replace_op(choose_op, hw.ArrayGetOp(create_array, choose_op.switch))


class ConvertYieldOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, yield_op: phs.YieldOp, rewriter: PatternRewriter):
        output_op = hw.OutputOp(yield_op.operands)
        rewriter.replace_op(yield_op, output_op)


class ConvertPhsToHWPass(ModulePass):
    name = "convert-phs-to-hw"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        MLIROptPass(
            executable="circt-opt", generic=True, arguments=("--map-arith-to-comb", "--allow-unregistered-dialect")
        ).apply(ctx, op)
        PatternRewriteWalker(ConvertPeOps(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertMuxes(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertChooseOps(), apply_recursively=False).rewrite_module(op)
        PatternRewriteWalker(ConvertYieldOps(), apply_recursively=False).rewrite_module(op)
