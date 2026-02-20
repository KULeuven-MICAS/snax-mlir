from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.dialects.builtin import IntegerType, ModuleOp, SymbolRefAttr
from xdsl.ir import Attribute, Operation, SSAValue, TypeAttribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects.hardfloat import (
    AddRecFnOp,
    Exception5,
    FnToRecFnOp,
    HardfloatOperation,
    MulRecFnOp,
    RecFnToFnOp,
    Rounding,
)


def get_suffix(op: HardfloatOperation) -> str:
    if op.int_width is None:
        return f"_s{op.sig_width.data}_e{op.exp_width.data}"
    else:
        return f"_s{op.sig_width.data}_e{op.exp_width.data}_i{op.int_width.data}"


def get_count(counts: dict[HWBlockSpec, int], hw_block: HWBlockSpec) -> int:
    if hw_block in counts:
        count = counts[hw_block]
    else:
        counts[hw_block] = 0
        count = 0
    # increment for next invocation
    counts[hw_block] = counts[hw_block] + 1
    return count


@dataclass(frozen=True)
class HWBlockSpec:
    """
    Represents an external HW module that can be called

    Makes it easier to wrangle the hw dialect.

    This helper allows one to easily define a certain piece of hardware declaratively, and then create
    both hw.instance ops and hw.module.external ops.
    """

    symbol_name: str
    in_ports: Sequence[str]
    in_types: Sequence[Attribute]
    out_ports: Sequence[str]
    out_types: Sequence[Attribute]

    @property
    def symbol_attr(self) -> SymbolRefAttr:
        return SymbolRefAttr(self.symbol_name)

    def instance(self, name: str, inputs: Sequence[SSAValue]) -> hw.InstanceOp:
        return hw.InstanceOp(
            name,
            self.symbol_attr,
            tuple(
                zip(self.in_ports, inputs, strict=True),
            ),
            tuple(
                zip(
                    self.out_ports,
                    [cast(TypeAttribute, out_type) for out_type in self.out_types],
                )
            ),
        )

    def module(self) -> hw.HWModuleExternOp:
        in_ports: list[hw.ModulePort] = []
        for p, t in zip(self.in_ports, self.in_types):
            in_ports.append(
                hw.ModulePort(builtin.StringAttr(p), cast(TypeAttribute, t), hw.DirectionAttr(hw.Direction.INPUT))
            )
        out_ports: list[hw.ModulePort] = []
        for p, t in zip(self.out_ports, self.out_types):
            out_ports.append(
                hw.ModulePort(builtin.StringAttr(p), cast(TypeAttribute, t), hw.DirectionAttr(hw.Direction.OUTPUT))
            )
        mod_type = hw.ModuleType(builtin.ArrayAttr([*in_ports, *out_ports]))
        return hw.HWModuleExternOp(builtin.StringAttr(self.symbol_name), mod_type)

    @staticmethod
    def from_op(op: HardfloatOperation) -> HWBlockSpec:
        suffix = get_suffix(op)
        symbol_name = op.get_chisel_name() + suffix
        # Input names
        input_names = ["a", "b", "c"][: len(op.operands)]
        input_types = [*op.operand_types]
        if Rounding() in op.traits:
            input_names.extend(["roundingMode", "detectTininess"])
            input_types.extend([IntegerType(3), IntegerType(1)])
        io_input_names = (*(f"io_{name}" for name in input_names),)
        # Output names
        output_names = ["out"]
        output_types = [*op.result_types]
        if Exception5() in op.traits:
            output_names.extend(["exceptionFlags"])
            output_types.extend([IntegerType(5)])
        io_output_names = (*(f"io_{name}" for name in output_names),)
        return HWBlockSpec(symbol_name, io_input_names, tuple(input_types), io_output_names, tuple(output_types))


@dataclass
class ConvertSimpleOps(RewritePattern):
    counts: dict[HWBlockSpec, int] = field(default_factory=dict[HWBlockSpec, int])

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: AddRecFnOp | MulRecFnOp | RecFnToFnOp | FnToRecFnOp,
        rewriter: PatternRewriter,
    ):
        match op:
            case AddRecFnOp():
                new_ops: list[Operation] = [
                    c0_i3 := hw.ConstantOp(0, 3),
                    c0_i1 := hw.ConstantOp(0, 1),
                ]
                input_vals = [*op.operands, c0_i3.result, c0_i1.result]
                hw_block = HWBlockSpec.from_op(op)
            case MulRecFnOp():
                new_ops: list[Operation] = [
                    c0_i3 := hw.ConstantOp(0, 3),
                    c0_i1 := hw.ConstantOp(0, 1),
                ]
                input_vals = [*op.operands, c0_i3.result, c0_i1.result]
                hw_block = HWBlockSpec.from_op(op)
            case RecFnToFnOp():
                new_ops: list[Operation] = []
                input_vals = op.operands
                hw_block = HWBlockSpec.from_op(op)
            case FnToRecFnOp():
                new_ops: list[Operation] = []
                input_vals = [*op.operands]
                hw_block = HWBlockSpec.from_op(op)
        symbol_name = hw_block.symbol_name
        count = get_count(self.counts, hw_block)
        instance_op = hw_block.instance(f"{symbol_name}_{count}", input_vals)
        new_ops.append(instance_op)
        rewriter.replace_op(op, new_ops, new_results=[instance_op.results[0]])


class ConvertHardfloatToHw(ModulePass):
    name = "convert-hardfloat-to-hw"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        counts: dict[HWBlockSpec, int] = {}
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertSimpleOps(counts)]),
            apply_recursively=False,
        ).rewrite_module(op)
        body = op.body.block
        assert body is not None
        for spec in sorted(counts.keys(), key=lambda spec: spec.symbol_name):
            body.add_op(spec.module())
