from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr
from xdsl.ir import Attribute, SSAValue, TypeAttribute
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import hardfloat


def get_suffix(op: hardfloat.HardfloatOperation) -> str:
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
            tuple(zip(self.in_ports, inputs)),
            tuple(zip(self.out_ports, [cast(TypeAttribute, out_type) for out_type in self.out_types])),
        )

    def module(self) -> hw.HWModuleExternOp:
        mod_type = hw.ModuleType(
            builtin.ArrayAttr(
                [
                    *(
                        hw.ModulePort(
                            builtin.StringAttr(p), cast(TypeAttribute, t), hw.DirectionAttr(hw.Direction.INPUT)
                        )
                        for p, t in zip(self.in_ports, self.in_types)
                    ),
                    *(
                        hw.ModulePort(
                            builtin.StringAttr(p), cast(TypeAttribute, t), hw.DirectionAttr(hw.Direction.OUTPUT)
                        )
                        for p, t in zip(self.out_ports, self.out_types)
                    ),
                ]
            )
        )
        return hw.HWModuleExternOp(builtin.StringAttr(self.symbol_name), mod_type)


@dataclass
class ConvertSimpleOps(RewritePattern):
    counts: dict[HWBlockSpec, int] = field(default_factory=dict[HWBlockSpec, int])

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: hardfloat.AddRecFnOp
        | hardfloat.MulRecFnOp
        | hardfloat.RecFnToFnOp
        | hardfloat.FnToRecFnOp
        | hardfloat.RecFnToInOp
        | hardfloat.InToRecFnOp,
        rewriter: PatternRewriter,
    ):
        suffix = get_suffix(op)
        match op:
            case hardfloat.AddRecFnOp():
                symbol_name = f"AddRecFN{suffix}"
            case hardfloat.MulRecFnOp():
                symbol_name = f"MulRecFN{suffix}"
            case hardfloat.RecFnToFnOp():
                symbol_name = f"RecFnToFnOp{suffix}"
            case hardfloat.FnToRecFnOp():
                symbol_name = f"FnToRecFnOp{suffix}"
            case hardfloat.RecFnToInOp():
                symbol_name = f"RecFnToInOp{suffix}"
            case hardfloat.InToRecFnOp():
                symbol_name = f"InToRecFnOp{suffix}"
        hw_block = HWBlockSpec(symbol_name, ("io_a",), op.operand_types, ("io_out",), op.result_types)
        count = get_count(self.counts, hw_block)
        instance_op = hw_block.instance(f"{symbol_name}_{count}", [*op.operands])
        rewriter.replace_op(op, instance_op, new_results=instance_op.results)


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
        # for i, (spec, count) in enumerate(counts.items()):
        #    print(f"{i}) {spec.symbol_name:20} : {count}")
