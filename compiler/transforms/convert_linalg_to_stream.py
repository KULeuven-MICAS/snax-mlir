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

from compiler.dialects import stream


@dataclass
class StreamifyGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.Generic, rewriter: PatternRewriter) -> None:
        # place guard for library calls ending in _stream
        if not op.library_call:
            return
        if op.library_call.data.endswith("_stream"):
            op.library_call = StringAttr(op.library_call.data[: -len("_stream")])
        else:
            return

        input_count = len(op.inputs)
        streamable_input_indices = tuple(
            (index, arg.type)
            for index, (i, arg) in enumerate(
                zip(op.inputs, op.body.block.args[:input_count])
            )
            if isinstance(i.type, ShapedType) and arg.uses
        )
        streamable_output_indices = tuple(
            (index, arg.type)
            for index, (o, arg) in enumerate(
                zip(op.outputs, op.body.block.args[input_count:])
            )
            if isinstance(o.type, ShapedType)
        )

        input_stream_types = tuple(
            stream.StreamType(el_type) for _, el_type in streamable_input_indices
        )
        output_stream_types = tuple(
            stream.StreamType(el_type) for _, el_type in streamable_output_indices
        )
        result_stream_types = tuple(
            stream.StreamType(el_type) for _, el_type in streamable_output_indices
        )

        patterns = ArrayAttr(
            indexing_map
            for index, _ in (*streamable_input_indices, *streamable_output_indices)
            if (indexing_map := op.indexing_maps.data[index])
        )

        streaming_region_op = stream.StreamingRegionOp(
            inputs=tuple(op.inputs[index] for index, _ in streamable_input_indices),
            outputs=tuple(op.outputs[index] for index, _ in streamable_output_indices),
            patterns=patterns,
            body=Region(Block(arg_types=input_stream_types + output_stream_types)),
            result_types=op.result_types,
            accelerator=op.library_call,
        )

        new_body = streaming_region_op.body.block

        new_inputs = list(op.inputs)
        for stream_index, (index, _) in enumerate(streamable_input_indices):
            new_inputs[index] = new_body.args[stream_index]

        rewriter.insert_op(
            (
                generic := stream.GenericOp(
                    new_inputs,
                    rewriter.move_region_contents_to_new_regions(op.body),
                    op.doc,
                    op.library_call,
                    result_stream_types,
                ),
                stream.YieldOp(generic.results[0]),
            ),
            InsertPoint.at_end(new_body),
        )

        # replace linalg yield with stream yield
        assert isinstance(yield_op := generic.body.block.last_op, linalg.YieldOp)
        rewriter.replace_op(yield_op, stream.YieldOp(yield_op.operands[0]))

        rewriter.replace_matched_op(streaming_region_op)


@dataclass(frozen=True)
class ConvertLinalgToStream(ModulePass):
    """
    Converts a linalg generic to a stream generic wrapped in
    a streaming region.
    """

    name = "convert-linalg-to-stream"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        PatternRewriteWalker(StreamifyGenericOpPattern()).rewrite_module(op)
