from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.mlir_opt import MLIROptPass

MLIR_PREPOC_FLAGS: tuple[tuple[str, ...], ...] = (
    (
        "--pass-pipeline=builtin.module(func.func("
        + ", ".join(
            [
                "tosa-to-linalg-named",
                "tosa-to-tensor",
                "tosa-to-scf",
                "tosa-to-linalg",
            ]
        )
        + "))",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
    ),
    (
        "--tosa-to-arith=include-apply-rescale",
        "--empty-tensor-to-alloc-tensor",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
    ),
    (
        "--test-linalg-transform-patterns=test-generalize-pad-tensor",
        "--linalg-generalize-named-ops",
        "--empty-tensor-to-alloc-tensor",
        "--one-shot-bufferize="
        + " ".join(
            [
                "bufferize-function-boundaries",
                "allow-return-allocs-from-loops",
                "function-boundary-type-conversion=identity-layout-map",
            ]
        ),
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
    ),
)


@dataclass(frozen=True)
class PreprocessPass(ModulePass):
    name = "preprocess"

    executable: str = field(default="mlir-opt")

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        for flags in MLIR_PREPOC_FLAGS:
            MLIROptPass(generic=False, arguments=flags).apply(ctx, op)
