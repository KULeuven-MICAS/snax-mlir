from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

from snaxc.dialects import phs
from snaxc.phs.hw_conversion import create_pe_array, get_pe_port_decl, get_switch_bitwidth
from snaxc.phs.template_spec import TemplateSpec


@dataclass(frozen=True)
class ConvertPeOps(RewritePattern):
    template_spec: TemplateSpec | None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, pe: phs.PEOp, rewriter: PatternRewriter):
        ports_attr = get_pe_port_decl(pe)
        mod_type = hw.ModuleType(ports_attr)
        hw_mod = hw.HWModuleOp(sym_name=pe.name_prop, module_type=mod_type, body=pe.body.clone(), visibility="private")
        new_ops = [hw_mod]
        if self.template_spec is None:
            rewriter.replace_op(pe, new_ops)
        else:
            hw_array = create_pe_array(pe, self.template_spec)
            new_ops.append(hw_array)
            rewriter.replace_op(pe, new_ops)

        # Fix type mismatch in switch type conversion
        data_opnd_idx = len(pe.data_operands())
        switch_idx = len(pe.get_switches())

        # Create insertion point
        block = hw_mod.regions[0].block
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

        # Replace yield_op with output_op
        yield_op = block.ops.last
        assert isinstance(yield_op, phs.YieldOp)
        output_op = hw.OutputOp(yield_op.operands)
        rewriter.replace_op(yield_op, output_op)


@dataclass(frozen=True)
class ConvertPEToHWPass(ModulePass):
    name = "convert-pe-to-hw"

    bounds: tuple[int, ...] | None = None

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # Fixme: At some point this template_spec should not be fixed!
        if self.bounds is not None:
            fixed_input_maps = (
                AffineMap.from_callable(lambda y: (y,)),
                AffineMap.from_callable(lambda y: (y,)),
                AffineMap.from_callable(lambda y: (y,)),
            )
            fixed_output_maps = (AffineMap.from_callable(lambda y: (y,)),)
            template_spec = TemplateSpec(
                input_maps=fixed_input_maps, output_maps=fixed_output_maps, template_bounds=self.bounds
            )
        else:
            template_spec = None
        PatternRewriteWalker(ConvertPeOps(template_spec=template_spec), apply_recursively=False).rewrite_module(op)
