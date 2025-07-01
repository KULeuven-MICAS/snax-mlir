from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.transforms.mlir_opt import MLIROptPass

from snaxc.transforms.frontend.remove_transpose_constants import RemoveTransposeConstants

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
        "--allow-unregistered-dialect",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
        "--allow-unregistered-dialect",
    ),
    (
        "--tosa-to-arith=include-apply-rescale",
        "--allow-unregistered-dialect",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
        "--allow-unregistered-dialect",
    ),
    (
        "--linalg-generalize-named-ops",
        "--allow-unregistered-dialect",
        "--canonicalize",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
        "--allow-unregistered-dialect",
    ),
)


@dataclass(frozen=True)
class PreprocessPass(ModulePass):
    name = "preprocess"

    executable: str = field(default="mlir-opt")

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for flags in MLIR_PREPOC_FLAGS:
            MLIROptPass(generic=True, arguments=flags).apply(ctx, op)

        PatternRewriteWalker(RemoveTransposeConstants(), apply_recursively=False).rewrite_module(op)
