from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.context import Context
from xdsl.dialects import builtin, hw
from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr
from xdsl.ir import Attribute, SSAValue, TypeAttribute
from xdsl.parser import ParseError, Parser
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.exceptions import DiagnosticException

from snaxc.dialects.hardfloat import (
    HardfloatOperation,
)


def get_suffix(op: HardfloatOperation) -> str:
    if op.int_width is None:
        return f"_s{op.sig_width.data}_e{op.exp_width.data}"
    else:
        return f"_i{op.int_width.data}_s{op.sig_width.data}_e{op.exp_width.data}"


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
        input_names = (*(f"io_{name}" for name in op.get_chisel_input_names()),)
        output_names = (*(f"io_{name}" for name in op.get_chisel_output_names()),)
        return HWBlockSpec(symbol_name, input_names, op.operand_types, output_names, op.result_types)


@dataclass
class ConvertHardfloatOps(RewritePattern):
    counts: dict[HWBlockSpec, int] = field(default_factory=dict[HWBlockSpec, int])

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: HardfloatOperation,
        rewriter: PatternRewriter,
    ):
        hw_block = HWBlockSpec.from_op(op)
        symbol_name = hw_block.symbol_name
        count = get_count(self.counts, hw_block)
        instance_op = hw_block.instance(f"{symbol_name}_{count}", op.operands)
        rewriter.replace_op(op, instance_op, new_results=instance_op.results)


class ConvertHardfloatToHw(ModulePass):
    name = "convert-hardfloat-to-hw"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        counts: dict[HWBlockSpec, int] = {}
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConvertHardfloatOps(counts)]),
            apply_recursively=False,
        ).rewrite_module(op)

        external_modules = False
        if external_modules:
            body = op.body.block
            assert body is not None
            for spec in sorted(counts.keys(), key=lambda spec: spec.symbol_name):
                body.add_op(spec.module())
        else:
            # Call EasyFloat to generate hw dialect for blocks and parse the output
            ops = ",".join([spec.symbol_name for spec in counts.keys()])
            mill_cmd = f"mill 'EasyFloat.run' --ops {ops} --format=hw"
            try:
                mill_process = subprocess.run(
                    mill_cmd,
                    cwd="/home/josse/kuleuven-easyfloat",
                    capture_output=True,
                    shell=True,
                    text=True,
                    check=True,
                )
                opt_process = subprocess.run(
                    ["circt-opt", "--strip-om"], input=mill_process.stdout, capture_output=True, text=True, check=True
                )
            except subprocess.CalledProcessError as e:
                if e.stdout:
                    print("\n[STDOUT CAPTURE]")
                    print(e.stdout)
                if e.stderr:
                    print("\n[STDERR CAPTURE]")
                    print(e.stderr)
                exit(0)
            # Get the stdout output
            stdout_output = opt_process.stdout
            parser = Parser(ctx, stdout_output)
            try:
                new_module = parser.parse_module()
            except ParseError as e:
                raise DiagnosticException("Error parsing EasyFloat output") from e
            body = op.body.block
            assert body is not None
            for easyfloat_op in new_module.ops:
                if not isinstance(easyfloat_op, hw.HWModuleOp):
                    continue
                if easyfloat_op.sym_name.data == "EasyFloatTop":
                    continue
                easyfloat_op.detach()
                body.add_op(easyfloat_op)
