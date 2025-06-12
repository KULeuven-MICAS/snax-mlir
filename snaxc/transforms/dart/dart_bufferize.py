from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import bufferization, builtin
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.dialects import dart


@dataclass
class BufferizeStreamingRegion(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.OperationOp, rewriter: PatternRewriter) -> None:
        # check for operands that need to be bufferized:
        operands_to_buffer = tuple(operand for operand in op.operands if isinstance(operand.type, builtin.TensorType))

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

        new_op = dart.OperationOp(
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
class DartBufferize(ModulePass):
    name = "dart-bufferize"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(BufferizeStreamingRegion()).rewrite_module(op)
