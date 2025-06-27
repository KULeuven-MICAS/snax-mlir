from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriteWalker,
)
from xdsl.transforms.mlir_opt import MLIROptPass

from snaxc.transforms.dart.dart_bufferize import BufferizeStreamingRegion, VerifyDartBufferization


@dataclass(frozen=True)
class SnaxBufferize(ModulePass):
    name = "snax-bufferize"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)

    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops allow-unknown-ops"
            + " function-boundary-type-conversion=identity-layout-map",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    mlir_canonicalization_pass = MLIROptPass(
        arguments=(
            "--canonicalize",
            "--cse",
            "--canonicalize",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(VerifyDartBufferization()).rewrite_module(op)
        self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(BufferizeStreamingRegion()).rewrite_module(op)
        self.mlir_canonicalization_pass.apply(ctx, op)
