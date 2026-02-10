import struct
from collections.abc import Sequence
from dataclasses import field
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, builtin, hw
from xdsl.ir import Operation, SSAValue, TypeAttribute, dataclass
from xdsl.irdl import isa
from xdsl.parser import AnyFloat, StringAttr, SymbolRefAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPattern


@dataclass(frozen=True)
class HWBLockSpec:
    """
    Represents an external HW module that can be called

    Makes it easier to wrangle CIRCTs hw dialect.

    This helper allows one to easily define a certain piece of hardware declaratively, and then create
    both hw.instance ops and hw.module.external ops.
    """

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
i1 = builtin.i1
i3 = builtin.IntegerType(3)
i5 = builtin.IntegerType(5)

_HW_BLOCKS = {
    arith.AddfOp: HWBLockSpec("AddRecFN", ("io_a", "io_b"), (i33, i33), ("io_out",), (i33,)),
    arith.MulfOp: HWBLockSpec("MulRecFN", ("io_a", "io_b"), (i33, i33), ("io_out",), (i33,)),
}

_recode = HWBLockSpec("RecFNFromFN", ("io_in",), (i32,), ("io_out",), (i33,))
_unrecode = HWBLockSpec("fNFromRecFN", ("io_in",), (i33,), ("io_out",), (i32,))
_fptosi = HWBLockSpec(
    "RecFNToIN", ("in", "roundingMode", "signedOut"), (i33, i3, i1), ("out", "intExceptionFlags"), (i32, i3)
)
_sitofp = HWBLockSpec(
    "INToRecFN",
    ("signedIn", "in", "roundingMode", "detectTininess"),
    (i1, i32, i3, i1),
    ("out", "exceptionFlags"),
    (i33, i5),
)


@dataclass
class ConvertFloatToHardFloat(RewritePattern):
    """
    Generic pattern to rewrite any arith op present in _HW_BLOCKS to its berkeley
    hardfloat module invocation.

    It keeps trak of all inserted modules in the internal `seen` set that can be
    used to insert the final set of external modules.
    """

    seen: set[HWBLockSpec] = field(default_factory=set)
    counter = 0

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, tuple(_HW_BLOCKS)):
            return

        spec = _HW_BLOCKS[type(op)]
        self.seen.add(spec)
        self.seen.add(_recode)
        self.seen.add(_unrecode)

        enc_a, a = ConvertFloatToHardFloat._build_input(op.lhs, f"lhs_{self.counter}")
        if op.lhs is op.rhs:
            enc_b: Sequence[Operation] = []
            b = a
        else:
            enc_b, b = ConvertFloatToHardFloat._build_input(op.rhs, f"rhs_{self.counter}")

        rewriter.replace_op(
            op,
            [
                *enc_a,
                *enc_b,
                actual_op := spec.instance(f"{op.name}_{self.counter}", [a, b]),
                res_recoded := _unrecode.instance(f"unrecoderr_{self.counter}", actual_op.results),
                ucc := builtin.UnrealizedConversionCastOp.get(res_recoded.results, [op.result.type]),
            ],
            ucc.results,
        )
        self.counter += 1

    @classmethod
    def _build_input(cls, arg: SSAValue, name: str) -> tuple[Sequence[Operation], SSAValue]:
        assert isa(arg.type, AnyFloat)
        int_t = builtin.IntegerType(arg.type.bitwidth)
        return (
            [
                ucc := builtin.UnrealizedConversionCastOp.get([arg], [int_t]),
                recode := _recode.instance(f"recoder_{name}", ucc.results),
            ],
            recode.results[0],
        )


class CancelUnrecodeRecode(RewritePattern):
    """
    Cancel pairs of unrecode->recode ops in the code.

    This will perform DCE on the "dead" instances as they are
    not generally pure and hence not caught by dce pass.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: hw.InstanceOp, rewriter: PatternRewriter, /):
        # only apply to recode operations:
        if op.module_name.string_value() != _recode.symbol_name:
            return
        op_owner = op.operands[0].owner
        # and if the op_owner is an Instance of the unrecode op
        if not isinstance(op_owner, hw.InstanceOp) or op_owner.module_name.string_value() != _unrecode.symbol_name:
            return

        # replace our results by the inputs of the unrecode op
        # inputs to the unrecode are already the recoded format
        # so we can skip the unrecode->recode loop entirely
        for res, inp in zip(op.results, op_owner.operands):
            res.replace_by(inp)
        # erase the recode op:
        rewriter.erase_op(op)

        # erase the unreocde op if no longer used
        if all(res.uses.get_length() == 0 for res in op_owner.results):
            rewriter.erase_op(op_owner)


class LowerArithBitcastOp(RewritePattern):
    """
    Convert an arith.bitcast op to precisely nothing
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.BitcastOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(
            op, ucc := builtin.UnrealizedConversionCastOp.get(op.operands, [op.result.type]), ucc.results
        )


@dataclass
class ConvertSiToFPOp(RewritePattern):
    """
    Convert an arith.sitofp op with a call to RecFNtoIN
    """

    seen: set[HWBLockSpec] = field(default_factory=set)
    counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SIToFPOp, rewriter: PatternRewriter, /):
        self.seen.add(_sitofp)
        self.seen.add(_unrecode)

        rewriter.replace_op(
            op,
            [
                rnd := arith.ConstantOp(builtin.IntegerAttr(0, i3)),
                tru := arith.ConstantOp(builtin.IntegerAttr(1, i1)),
                fals := arith.ConstantOp(builtin.IntegerAttr(0, i1)),
                cast := _sitofp.instance(f"brrrr_{self.counter}", [tru.result, op.input, rnd.result, fals.result]),
                i32out := _unrecode.instance(f"unrecode_brrrr_{self.counter}", [cast.results[0]]),
                ucc := builtin.UnrealizedConversionCastOp.get(i32out.results, [op.result.type]),
            ],
            ucc.results,
        )
        self.counter += 1


class EncodeFloatConstantsPattern(RewritePattern):
    """
    Encode floating-point constants as i32 patterns
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.ConstantOp, rewriter: PatternRewriter, /):
        # this only works on 32-bit floats right now, can't be arsed
        if op.value.type != builtin.f32:
            return
        float_val = float(cast(int | float, op.value.value.data))
        int_val = cast(int, struct.unpack("i", struct.pack("f", float_val))[0])

        rewriter.replace_op(
            op,
            [
                i32val := arith.ConstantOp(builtin.IntegerAttr(int_val, i32)),
                ucc := builtin.UnrealizedConversionCastOp.get(i32val.results, [builtin.f32]),
            ],
            ucc.results,
        )


class PhsConvertFloatToHardfloatPass(ModulePass):
    """
    Convert arith ops to barkeley hardfloat modules.
    """

    name = "phs-convert-float-to-hardfloat"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        seen: set[HWBLockSpec] = set()

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertFloatToHardFloat(seen),
                    ConvertSiToFPOp(seen),
                    EncodeFloatConstantsPattern(),
                    # we need to remove ucc as they interfer with the cancellation pass
                    ReconcileUnrealizedCastsPattern(),
                    CancelUnrecodeRecode(),
                    LowerArithBitcastOp(),
                ]
            )
        ).rewrite_module(op)

        body = op.body.block
        assert body is not None
        for spec in sorted(seen, key=lambda spec: spec.symbol_name):
            body.add_op(spec.module())
