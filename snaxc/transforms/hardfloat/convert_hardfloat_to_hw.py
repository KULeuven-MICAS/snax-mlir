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
    seen: set[HWBlockSpec] = field(default_factory=set[HWBlockSpec])
    counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: hardfloat.AddRecFnOp | hardfloat.MulRecFnOp | hardfloat.RecFnToFnOp | hardfloat.FnToRecFnOp,
        rewriter: PatternRewriter,
    ):
        suffix = get_suffix(op)
        match op:
            case hardfloat.AddRecFnOp():
                instance_name = f"AddRecFN{suffix}"
            case hardfloat.MulRecFnOp():
                instance_name = f"MulRecFN{suffix}"
            case hardfloat.RecFnToFnOp():
                instance_name = f"RecFnToFnOp{suffix}"
            case _:  # isinstance(op, hardfloat.FnToRecFnOp):
                instance_name = f"FnToRecFnOp{suffix}"
        add_block = HWBlockSpec(instance_name, ("io_a",), op.operand_types, ("io_out",), op.result_types)
        self.seen.add(add_block)
        instance_op = add_block.instance(f"{instance_name}_{self.counter}", [*op.operands])
        self.counter += 1
        rewriter.replace_op(op, instance_op, new_results=instance_op.results)


class ConvertHardfloatToHw(ModulePass):
    name = "convert-hardfloat-to-hw"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        seen: set[HWBlockSpec] = set()
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertSimpleOps(seen)]),
            apply_recursively=False,
        ).rewrite_module(op)
        body = op.body.block
        assert body is not None
        for spec in sorted(seen, key=lambda spec: spec.symbol_name):
            body.add_op(spec.module())
