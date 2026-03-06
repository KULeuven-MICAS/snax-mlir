import itertools
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.ir import SSAValue, TypeAttribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

from snaxc.phs.hw_conversion import create_shaped_hw_array, get_from_shaped_hw_array, get_shaped_hw_array_shape


class ScalarizeHwModules(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: hw.HWModuleOp, rewriter: PatternRewriter):
        # Only work on public modules
        if op.sym_visibility is not None:
            return
        block = op.body.block

        # ── 1. Build new port list and replace array input args ──────────
        new_ports: list[hw.ModulePort] = []
        insert_offset: int = 0
        for port in op.module_type.ports:
            if port.dir.data != hw.Direction.INPUT or not isa(port.type, hw.ArrayType):
                new_ports.append(port)
                insert_offset += 1
                continue

            shape, el_type = get_shaped_hw_array_shape(port.type)
            flat_size = 1
            for s in shape:
                flat_size *= s

            # Insert scalar block args in place of the array arg
            old_arg = block.args[insert_offset]
            new_scalar_args: list[SSAValue] = []
            for i in range(flat_size):
                new_scalar_args.append(rewriter.insert_block_argument(block, insert_offset + i, el_type))

            # Reconstruct the array from scalars at top of block,
            # replace all uses of old array arg, let canonicalization clean up
            reconst_ops, reconst_val = create_shaped_hw_array(new_scalar_args, tuple(shape))
            rewriter.insert_op(reconst_ops, InsertPoint.at_start(block))
            old_arg.replace_all_uses_with(reconst_val)
            rewriter.erase_block_argument(old_arg)

            for i in range(flat_size):
                new_ports.append(
                    hw.ModulePort(
                        builtin.StringAttr(f"{port.port_name.data}_{i}"),
                        cast(TypeAttribute, el_type),
                        hw.DirectionAttr(data=hw.Direction.INPUT),
                    )
                )
            insert_offset += flat_size
        # ── 2. Scalarize output ports in hw.output ───────────────────────
        output_op = block.last_op
        assert isinstance(output_op, hw.OutputOp)

        new_output_operands: list[SSAValue] = []
        new_output_ports: list[hw.ModulePort] = []

        for operand, port in zip(
            output_op.operands, [p for p in op.module_type.ports if p.dir.data == hw.Direction.OUTPUT]
        ):
            if not isa(operand.type, hw.ArrayType):
                new_output_operands.append(operand)
                new_output_ports.append(port)
                continue

            shape, el_type = get_shaped_hw_array_shape(operand.type)

            for index in itertools.product(*[range(s) for s in shape]):
                get_ops, scalar_val = get_from_shaped_hw_array(cast(SSAValue[hw.ArrayType], operand), index)
                rewriter.insert_op(get_ops, InsertPoint(block, insert_before=output_op))
                new_output_operands.append(scalar_val)
                new_output_ports.append(
                    hw.ModulePort(
                        builtin.StringAttr(f"{port.port_name.data}_{'_'.join(str(i) for i in index)}"),
                        cast(TypeAttribute, el_type),
                        hw.DirectionAttr(data=hw.Direction.OUTPUT),
                    )
                )

        rewriter.replace_op(output_op, hw.OutputOp(new_output_operands))

        # ── 3. Update module type ─────────────────────────────────────────
        all_new_ports = [p for p in new_ports if p.dir.data != hw.Direction.OUTPUT] + new_output_ports
        op.module_type = hw.ModuleType(builtin.ArrayAttr(all_new_ports))


class HwScalarizePublicModulesPass(ModulePass):
    name = "hw-scalarize-public-modules"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ScalarizeHwModules(), apply_recursively=False).rewrite_module(op)
