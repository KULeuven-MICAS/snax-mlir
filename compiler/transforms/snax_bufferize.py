from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.mlir_opt import MLIROptPass


@dataclass(frozen=True)
class SnaxBufferize(ModulePass):
    name = "snax-bufferize"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)

    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            "--one-shot-bufferize=bufferize-function-boundaries allow-return-allocs-from-loops allow-unknown-ops"
            + " function-boundary-type-conversion=identity-layout-map",
            "--buffer-deallocation-pipeline",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        self.mlir_bufferization_pass.apply(ctx, op)
