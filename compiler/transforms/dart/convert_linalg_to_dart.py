from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import linalg
from xdsl.dialects.builtin import ArrayAttr, ModuleOp, ShapedType, StringAttr
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects import dart


@dataclass
class StreamifyGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.GenericOp, rewriter: PatternRewriter
    ) -> None:
        # place guard for library calls ending in _stream
        if not op.library_call:
            return
        if op.library_call.data.endswith("_stream"):
            op.library_call = StringAttr(op.library_call.data[: -len("_stream")])
        else:
            return

        # find streamable operands: shaped operands of the generic op
        input_count = len(op.inputs)
        streamable_input_indices = tuple(
            (index, arg.type)
            for index, (i, arg) in enumerate(
                zip(op.inputs, op.body.block.args[:input_count])
            )
            if isinstance(i.type, ShapedType) and arg.uses
        )
        streamable_output_indices = tuple(
            (index + len(op.inputs), arg.type)
            for index, (o, arg) in enumerate(
                zip(op.outputs, op.body.block.args[input_count:])
            )
            if isinstance(o.type, ShapedType)
        )

        # create new stream.stream operand and result types
        input_stream_types = tuple(
            dart.StreamType(el_type) for _, el_type in streamable_input_indices
        )
        result_stream_types = tuple(
            dart.StreamType(el_type) for _, el_type in streamable_output_indices
        )

        # copy patterns from generic op
        patterns = ArrayAttr(
            indexing_map
            for index, _ in (*streamable_input_indices, *streamable_output_indices)
            if (indexing_map := op.indexing_maps.data[index])
        )

        # create the streaming region to wrap around the stream.generic
        streaming_region_op = dart.OperationOp(
            inputs=tuple(op.operands[index] for index, _ in streamable_input_indices),
            outputs=tuple(op.operands[index] for index, _ in streamable_output_indices),
            patterns=patterns,
            body=Region(Block(arg_types=input_stream_types + result_stream_types)),
            result_types=op.result_types,
            accelerator=op.library_call,
        )

        new_body = streaming_region_op.body.block

        # construct new inputs for the stream.generic
        new_inputs = list(op.inputs)
        for stream_index, (index, _) in enumerate(streamable_input_indices):
            new_inputs[index] = new_body.args[stream_index]

        # create stream.generic based on the linal.generic and put inside streaming region
        rewriter.insert_op(
            (
                generic := dart.GenericOp(
                    new_inputs,
                    rewriter.move_region_contents_to_new_regions(op.body),
                    op.doc,
                    op.library_call,
                    result_stream_types,
                ),
                dart.YieldOp(generic.results[0]),
            ),
            InsertPoint.at_end(new_body),
        )

        # replace linalg yield with stream yield
        assert isinstance(yield_op := generic.body.block.last_op, linalg.YieldOp)
        rewriter.replace_op(yield_op, dart.YieldOp(yield_op.operands[0]))

        rewriter.replace_matched_op(streaming_region_op)


@dataclass(frozen=True)
class ConvertLinalgToDart(ModulePass):
    """
    Converts a linalg generic to a dart generic wrapped in
    a dart operation.
    """

    name = "convert-linalg-to-dart"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(StreamifyGenericOpPattern()).rewrite_module(op)
