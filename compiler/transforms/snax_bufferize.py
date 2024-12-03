from dataclasses import dataclass, field

from xdsl.context import MLContext
from xdsl.dialects import bufferization, builtin
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.mlir_opt import MLIROptPass

from compiler.dialects import stream


@dataclass
class BufferizeStreamingRegion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: stream.StreamingRegionOp, rewriter: PatternRewriter
    ) -> None:
        # check if for operands that need to be bufferized:
        operands_to_buffer = tuple(
            operand
            for operand in op.operands
            if isinstance(operand.type, builtin.TensorType)
        )

        # if not tensor operands, return
        if not operands_to_buffer:
            return

        # for every unique input, make sure the tensor is the result
        # of a to_tensor operation and store the original memref
        tensor_to_memrefs: dict[SSAValue, SSAValue] = {}

        for operand in set(operands_to_buffer):
            if not isinstance(operand, OpResult):
                return
            if not isinstance(to_tensor_op := operand.op, bufferization.ToTensorOp):
                return
            tensor_to_memrefs[operand] = to_tensor_op.memref

        new_op = stream.StreamingRegionOp(
            inputs=[tensor_to_memrefs[input] for input in op.inputs],
            outputs=[tensor_to_memrefs[output] for output in op.outputs],
            patterns=op.patterns,
            body=rewriter.move_region_contents_to_new_regions(op.body),
            accelerator=op.accelerator,
        )

        # for every output, create a bufferization.to_tensor op
        memref_to_tensors: dict[SSAValue, Operation] = {}
        new_results: tuple[SSAValue, ...] = ()

        for output in new_op.outputs:
            to_tensor_op = bufferization.ToTensorOp(output, restrict=True)
            memref_to_tensors[output] = to_tensor_op
            new_results += to_tensor_op.results

        # replace the old operation
        rewriter.replace_matched_op(
            (new_op,) + tuple(memref_to_tensors.values()),
            new_results,
        )


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

    mlir_canonicalization_pass = MLIROptPass(
        arguments=(
            "--canonicalize",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
            "--allow-unregistered-dialect",
        )
    )

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(BufferizeStreamingRegion()).rewrite_module(op)
        self.mlir_canonicalization_pass.apply(ctx, op)
