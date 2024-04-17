from dataclasses import dataclass

from xdsl.dialects import builtin
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.dialects import acc


@dataclass
class InsertAcceleratorOpPattern(RewritePattern):
    acc_op: acc.AcceleratorOp

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter):
        t = op.get_trait(SymbolTable)
        assert t is not None
        t.insert_or_update(op, self.acc_op)


@dataclass(frozen=True)
class InsertAccOp(ModulePass):
    name = "insert-acc-op"

    accelerator: str  # accelerator name in registry

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # Access registry to get the accelerator interface
        acc_info = AcceleratorRegistry().get_acc_info(self.accelerator)
        # With the interface, generate an appropriate acc op
        acc_op = acc_info().generate_acc_op()
        # Use the get
        PatternRewriteWalker(InsertAcceleratorOpPattern(acc_op)).rewrite_module(op)
