from collections.abc import Sequence
from dataclasses import field
from typing import Type

from xdsl.context import Context
from xdsl.dialects import arith, builtin, hw
from xdsl.ir import Operation, SSAValue, TypeAttribute, dataclass
from xdsl.irdl import isa
from xdsl.parser import AnyFloat, StringAttr, SymbolRefAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern
from xdsl.rewriter import InsertPoint


@dataclass(frozen=True)
class HWBLockSpec:
    symbol_name: str
    in_ports: tuple[str, ...]
    in_types: tuple[TypeAttribute, ...]
    out_ports: tuple[str, ...]
    out_types: tuple[TypeAttribute, ...]

    @property
    def symbol_attr(self) -> SymbolRefAttr:
        return SymbolRefAttr(self.symbol_name)

    def instance(self, name: str, inputs: Sequence[SSAValue]) -> hw.InstanceOp:
        return hw.InstanceOp(
            name,
            self.symbol_attr,
            tuple(zip(self.in_ports, inputs)),
            tuple(zip(self.out_ports, self.out_types)),
        )

    def module(self) -> hw.HWModuleExternOp:
        mod_type = hw.ModuleType(
            builtin.ArrayAttr(
                [
                    *(
                        hw.ModulePort(builtin.StringAttr(p), t, hw.DirectionAttr(hw.Direction.INPUT))
                        for p, t in zip(self.in_ports, self.in_types)
                    ),
                    *(
                        hw.ModulePort(builtin.StringAttr(p), t, hw.DirectionAttr(hw.Direction.OUTPUT))
                        for p, t in zip(self.out_ports, self.out_types)
                    ),
                ]
            )
        )
        return hw.HWModuleExternOp(StringAttr(self.symbol_name), mod_type)


i33 = builtin.IntegerType(33)
i32 = builtin.i32

_HW_BLOCKS = {
    "arith.addf": HWBLockSpec("AddRecFN", ("io_a", "io_b"), (i33, i33), ("io_out",), (i33,)),
    "arith.mulf": HWBLockSpec("MulRecFN", ("io_a", "io_b"), (i33, i33), ("io_out",), (i33,)),
}

_recode = HWBLockSpec("RecFNFromFN", ("io_in",), (i32,), ("io_out",), (i33,))
_unrecode = HWBLockSpec("fNFromRecFN", ("io_in",), (i33,), ("io_out",), (i32,))


def _build_input(arg: SSAValue) -> tuple[Sequence[Operation], SSAValue]:
    assert isa(arg.type, AnyFloat)
    int_t = builtin.IntegerType(arg.type.bitwidth)
    return (
        [
            ucc := builtin.UnrealizedConversionCastOp.get([arg], [int_t]),
            recode := _recode.instance("recoderrr", ucc.results),
        ],
        recode.results[0],
    )


@dataclass
class ConvertFloatToHardFloat(RewritePattern):
    seen: set[HWBLockSpec] = field(default_factory=set)

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, (arith.AddfOp, arith.MulfOp)):
            return

        assert op.name in _HW_BLOCKS

        spec = _HW_BLOCKS[op.name]
        self.seen.add(spec)
        self.seen.add(_recode)
        self.seen.add(_unrecode)

        enc_a, a = _build_input(op.lhs)
        enc_b, b = _build_input(op.rhs)

        rewriter.replace_op(
            op,
            [
                *enc_a,
                *enc_b,
                actual_op := spec.instance(op.name, [a, b]),
                res_recoded := _unrecode.instance("unrecoderr", actual_op.results),
                ucc := builtin.UnrealizedConversionCastOp.get(res_recoded.results, [op.result.type]),
            ],
            ucc.results,
        )


class PhsConvertFloatToHardfloatPass(ModulePass):
    name = "phs-convert-float-to-hardfloat"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(pattern := ConvertFloatToHardFloat(), apply_recursively=False).rewrite_module(op)

        body = op.body.block
        assert body is not None
        for spec in sorted(pattern.seen, key=lambda spec: spec.symbol_name):
            body.add_op(spec.module())
