from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from snaxc.dialects import phs
from snaxc.phs.hw_conversion import get_pe_port_decl, get_switch_bitwidth


class ConvertPeOps(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, pe: phs.PEOp, rewriter: PatternRewriter):
        ports_attr = get_pe_port_decl(pe)
        mod_type = hw.ModuleType(ports_attr)
        new_op = hw.HWModuleOp(sym_name=pe.name_prop, module_type=mod_type, body=pe.body.clone(), visibility="private")
        rewriter.replace_op(pe, new_op)

        # Fix type mismatch in switch type conversion
        data_opnd_idx = len(pe.data_operands())
        switch_idx = len(pe.get_switches())

        # Create insertion point
        block = new_op.regions[0].block
        assert block.ops.first is not None
        ip = InsertPoint(block=block, insert_before=block.ops.first)

        # Insert
        for block_arg in block.args[data_opnd_idx : data_opnd_idx + switch_idx]:
            bitwidth = get_switch_bitwidth(block_arg)
            block_arg_index = block_arg.index
            ssaval = block.insert_arg(builtin.IntegerType(bitwidth), block_arg_index)
            op, res = builtin.UnrealizedConversionCastOp.cast_one(ssaval, builtin.IndexType())
            rewriter.replace_all_uses_with(block_arg, res)
            rewriter.insert_op(op, insertion_point=ip)
            block.erase_arg(block_arg)

        yield_op = block.ops.last
        assert isinstance(yield_op, phs.YieldOp)
        output_op = hw.OutputOp(yield_op.operands)
        rewriter.replace_op(yield_op, output_op)


class ConvertPEToHWPass(ModulePass):
    name = "convert-pe-to-hw"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertPeOps(), apply_recursively=False).rewrite_module(op)
