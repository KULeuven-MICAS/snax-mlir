from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.transforms.mlir_opt import MLIROptPass

MLIR_POSTPROC_FLAGS: tuple[str, ...] = (
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--lower-affine",
    "--canonicalize",
    "--cse",
    "--convert-math-to-llvm",
    "--llvm-request-c-wrappers",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-index-to-llvm=index-bitwidth={index_bitwidth}",
    "--convert-cf-to-llvm=index-bitwidth={index_bitwidth}",
    "--convert-arith-to-llvm=index-bitwidth={index_bitwidth}",
    "--convert-func-to-llvm=index-bitwidth={index_bitwidth}",
    "--finalize-memref-to-llvm=use-generic-functions index-bitwidth={index_bitwidth}",
    "--canonicalize",
    "--reconcile-unrealized-casts",
    "--mlir-print-local-scope",
    "--mlir-print-op-generic",
)


@dataclass(frozen=True)
class PostprocessPass(ModulePass):
    name = "postprocess"

    executable: str = field(default="mlir-opt")
    index_bitwidth: int = field(default=32)

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        flags = tuple(
            flag.format(index_bitwidth=self.index_bitwidth)
            for flag in MLIR_POSTPROC_FLAGS
        )
        # Temporarily allow unregistered ops
        allow_unregistered_old, ctx.allow_unregistered = ctx.allow_unregistered, True
        MLIROptPass(generic=True, arguments=flags).apply(ctx, op)
        # Reset unregistered ops value
        ctx.allow_unregistered = allow_unregistered_old
