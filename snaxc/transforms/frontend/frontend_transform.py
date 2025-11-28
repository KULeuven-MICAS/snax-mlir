from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.transform import NamedSequenceOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.transforms.mlir_opt import MLIROptPass
from xdsl.transforms.test_transform_dialect_erase_schedule import EraseTransformNamedSequenceOps

MLIR_FLAGS: tuple[tuple[str, ...], ...] = (
    (
        "--transform-interpreter",
        "--mlir-print-op-generic",
        "--mlir-print-local-scope",
    ),
)


@dataclass(frozen=True)
class FrontendTransformPass(ModulePass):
    name = "frontend-transform"

    executable: str = field(default="mlir-opt")

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # only apply the pass if the module contians a transform script
        if not any(isinstance(operation, NamedSequenceOp) for operation in op.ops):
            return
        for flags in MLIR_FLAGS:
            MLIROptPass(generic=True, arguments=flags).apply(ctx, op)
        PatternRewriteWalker(EraseTransformNamedSequenceOps(), apply_recursively=False).rewrite_module(op)
