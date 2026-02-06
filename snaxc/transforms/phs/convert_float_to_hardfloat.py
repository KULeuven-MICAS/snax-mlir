from xdsl.context import Context
from xdsl.dialects import arith, builtin, hw
from xdsl.parser import StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern


class ConvertFloatToHardFloat(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.AddfOp, rewriter: PatternRewriter, /):
        int_t = builtin.IntegerType(op.result.type.bitwidth)
        module = builtin.SymbolRefAttr("AddRecFN")
        args = ("io_a", "io_b")

        rewriter.replace_op(
            op,
            [
                ival := builtin.UnrealizedConversionCastOp.get(op.operands, [int_t, int_t]),
                hwc := hw.InstanceOp(op.name, module, list(zip(args, ival.results)), [("io_out", int_t)]),
                res := builtin.UnrealizedConversionCastOp.get(hwc.outputs, [op.result.type]),
            ],
            res.results,
        )


class PhsConvertFloatToHardfloatPass(ModulePass):
    name = "phs-convert-float-to-hardfloat"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(ConvertFloatToHardFloat(), apply_recursively=False).rewrite_module(op)
        int_t = builtin.IntegerType(32)
        module = builtin.SymbolRefAttr("AddRecFN")
        mod_type = hw.ModuleType(
            builtin.ArrayAttr(
                [
                    hw.ModulePort(builtin.StringAttr("io_a"), int_t, hw.DirectionAttr(hw.Direction.INPUT)),
                    hw.ModulePort(builtin.StringAttr("io_b"), int_t, hw.DirectionAttr(hw.Direction.INPUT)),
                    hw.ModulePort(builtin.StringAttr("io_out"), int_t, hw.DirectionAttr(hw.Direction.OUTPUT)),
                ]
            )
        )
        hw_extern = hw.HWModuleExternOp(StringAttr(module.string_value()), mod_type)
        assert op.body.block.ops.first is not None
        op.body.block.insert_op_before(hw_extern, op.body.block.ops.first)
