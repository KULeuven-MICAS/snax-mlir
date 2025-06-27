from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import linalg
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ModuleOp,
    ShapedType,
    StringAttr,
    TensorType,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import Block, BlockArgument, OpResult, Region, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

from snaxc.dialects import dart
from snaxc.dialects.kernel import AddOp


@dataclass
class StreamifyGenericOpPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp, rewriter: PatternRewriter) -> None:
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
            for index, (i, arg) in enumerate(zip(op.inputs, op.body.block.args[:input_count]))
            if isinstance(i.type, ShapedType) and arg.uses
        )
        streamable_output_indices = tuple(
            (index + len(op.inputs), arg.type)
            for index, (o, arg) in enumerate(zip(op.outputs, op.body.block.args[input_count:]))
            if isinstance(o.type, ShapedType)
        )

        # create new stream.stream operand and result types
        input_stream_types = tuple(dart.StreamType(el_type) for _, el_type in streamable_input_indices)
        result_stream_types = tuple(dart.StreamType(el_type) for _, el_type in streamable_output_indices)

        # copy patterns from generic op
        patterns = ArrayAttr(
            indexing_map
            for index, _ in (*streamable_input_indices, *streamable_output_indices)
            if (indexing_map := op.indexing_maps.data[index])
        )

        # if outputs isn't an empty tensor, explicit add should be added
        outputs: list[SSAValue] = []
        outputs_to_add: list[SSAValue] = []
        for output in (op.operands[index] for index, _ in streamable_output_indices):
            if isinstance(output, OpResult) and isinstance(output.op, EmptyOp):
                outputs.append(output)
            # replace with an empty tensor
            empty = EmptyOp([], tensor_type=output.type)
            rewriter.insert_op(empty, InsertPoint.before(op))
            outputs.append(empty.tensor)
            outputs_to_add.append(output)

        # create the streaming region to wrap around the stream.generic
        streaming_region_op = dart.OperationOp(
            inputs=tuple(op.operands[index] for index, _ in streamable_input_indices),
            outputs=tuple(outputs),
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

        # Add explicit adds for outputs that were not empty tensors
        for output in outputs_to_add:
            assert len(streaming_region_op.results) == 1
            output_idx = 0

            # add body for outputs:
            arg_types = [result_stream_types[output_idx].element_type] * 3
            stream_arg_types = [result_stream_types[output_idx]] * 3

            # generic body:
            @Builder.implicit_region(arg_types)
            def generic_region(args: tuple[BlockArgument, ...]) -> None:
                result = AddOp(operands=[args[0], args[1]], result_types=[args[2].type])
                dart.YieldOp(result)

            assert streaming_region_op.accelerator is not None

            @Builder.implicit_region(stream_arg_types)
            def dart_region(args: tuple[BlockArgument, ...]) -> None:
                result = dart.GenericOp(
                    inputs=[args[0], args[1]],
                    body=generic_region,
                    library_call=streaming_region_op.accelerator,
                    result_types=[args[2].type],
                )
                dart.YieldOp(result)

            if isinstance(output, OpResult) and isinstance(output.op, ConstantOp):
                empty = EmptyOp([], output.type)
                assert isa(output.type, TensorType)
                add_op = dart.OperationOp(
                    [streaming_region_op.results[0], output],
                    [empty],
                    ArrayAttr([AffineMapAttr(AffineMap.identity(output.type.get_num_dims()))] * 3),
                    body=dart_region,
                    result_types=[output.type],
                    accelerator=streaming_region_op.accelerator,
                )
                rewriter.insert_op([empty, add_op], InsertPoint.after(streaming_region_op))
                streaming_region_op.results[output_idx].replace_by_if(
                    add_op.results[0], lambda use: use.operation is not add_op
                )
            else:
                raise NotImplementedError("currently unsupported")


@dataclass(frozen=True)
class ConvertLinalgToDart(ModulePass):
    """
    Converts a linalg generic to a dart generic wrapped in
    a dart operation.
    """

    name = "convert-linalg-to-dart"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(StreamifyGenericOpPattern()).rewrite_module(op)
