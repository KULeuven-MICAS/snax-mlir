from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import bufferization
from xdsl.dialects.builtin import MemRefType, ModuleOp, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.core import OpResult
from xdsl.parser import UnitAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import stream


@dataclass
class BufferizeStreamingRegion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: stream.StreamingRegionOp, rewriter: PatternRewriter
    ) -> None:
        # check if for operands that need to be bufferized:
        operands_to_buffer = tuple(
            operand for operand in op.operands if isinstance(operand.type, TensorType)
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
            (new_op,)
            + tuple(memref_to_tensors.values()),
            new_results,
        )


@dataclass(frozen=True)
class StreamBufferize(ModulePass):
    """
    Bufferizes operations from the stream dialect.
    """

    name = "stream-bufferize"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(BufferizeStreamingRegion()).rewrite_module(op)
