from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Block, BlockArgument, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from snaxc.dialects import dart
from snaxc.dialects.kernel import AddOp


def pattern_is_broadcast(map: AffineMap):
    if not map.num_dims > len(map.results):
        return False
    for result in map.results:
        if not isinstance(result, AffineDimExpr):
            return False
    return True


@dataclass
class FuseElementwisePattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.OperationOp, rewriter: PatternRewriter):
        # only ops with 1 tensor result supported for now
        if len(op.results) != 1:
            return

        # get result and result_type
        result = op.results[0]
        assert isinstance(result.type, builtin.TensorType)

        # check if the result has exactly one use
        if result.uses.get_length() != 1:
            return

        user_op = list(result.uses)[0].operation

        # must be another streamingregion operation
        if not isinstance(user_op, dart.OperationOp):
            return

        # user op must be elementwise: all indexing maps must be identity maps
        for pattern in user_op.patterns:
            # Exception: broadcast patterns are supported for add
            if pattern_is_broadcast(pattern.data):
                assert isinstance(generic := user_op.body.block.first_op, dart.GenericOp)
                if not isinstance(generic.body.block.first_op, AddOp):
                    return
            elif pattern.data != AffineMap.identity(pattern.data.num_dims):
                return

        # now we can fuse!

        # find obliteration index:
        # after fusing regions, one output streamer from producer and one input
        # streamer from consumer will join and get obliterated. for the producer,
        # this is the last index (single output), for the consumer, we will find
        # this index here:
        obliterating_index = None
        for i, input in enumerate(user_op.inputs):
            if input is result:
                obliterating_index = i
        assert obliterating_index is not None

        # create a new streaming region op
        # new inputs: all inputs of both regions except for the fused operand
        new_inputs = op.inputs + tuple(i for i in user_op.inputs if i is not result)
        # new outputs: only outputs of consumer region
        new_outputs = user_op.outputs

        # patterns for producer region (all except fused operand = output):
        patterns = op.patterns.data[:-1]

        # add patterns for consumer region:
        num_dims = op.patterns.data[0].data.num_dims
        for p, operand in zip(user_op.patterns, user_op.operands):
            # obliterate pattern on which we fuse
            if operand is result:
                continue
            # create pattern with increased nb of dims
            patterns += (builtin.AffineMapAttr(AffineMap(num_dims, 0, p.data.results)),)
        patterns = builtin.ArrayAttr(patterns)

        # build arg types for body

        # first op arg types
        arg_types = tuple(arg.type for arg in op.body.block.args[:-1])
        # second op arg types
        arg_types += tuple(
            arg.type for arg, operand in zip(user_op.body.block.args, user_op.operands) if operand is not result
        )

        streaming_region_op = dart.OperationOp(
            new_inputs,
            new_outputs,
            patterns,
            Region(Block(arg_types=arg_types)),
            accelerator=op.accelerator,
            result_types=user_op.result_types,
        )

        producer_generic = None
        for o in op.body.block.ops:
            # make copies of generic ops from producer region
            if isinstance(o, dart.GenericOp):
                rewriter.insert_op(
                    producer_generic := dart.GenericOp(
                        inputs=tuple(
                            streaming_region_op.body.block.args[i.index]
                            if isinstance(i, BlockArgument) and isinstance(i.type, dart.StreamType)
                            else i
                            for i in o.inputs
                        ),
                        body=rewriter.move_region_contents_to_new_regions(o.body),
                        doc=o.doc,
                        library_call=o.library_call,
                        result_types=o.result_types,
                    ),
                    InsertPoint.at_end(streaming_region_op.body.block),
                )
                for old_result, new_result in zip(o.results, producer_generic.results):
                    old_result.replace_all_uses_with(new_result)
            # do not use yield op from producer region
            elif isinstance(o, dart.YieldOp):
                continue
            else:
                raise RuntimeError("Can only fuse with generic bodies")
        assert producer_generic is not None

        consumer_generic = None
        for o in user_op.body.block.ops:
            # make copies of generic ops from consumer region
            if isinstance(o, dart.GenericOp):
                rewriter.insert_op(
                    consumer_generic := dart.GenericOp(
                        inputs=tuple(
                            producer_generic.results[0]
                            if index == obliterating_index
                            else streaming_region_op.body.block.args[index + len(op.inputs) - 1]
                            for index in range(len(o.inputs))
                        ),
                        body=rewriter.move_region_contents_to_new_regions(o.body),
                        doc=o.doc,
                        library_call=o.library_call,
                        result_types=o.result_types,
                    ),
                    InsertPoint.at_end(streaming_region_op.body.block),
                )
            elif isinstance(o, dart.YieldOp):
                assert consumer_generic
                rewriter.insert_op(
                    dart.YieldOp(consumer_generic.results[0]),
                    InsertPoint.at_end(streaming_region_op.body.block),
                )

            else:
                raise RuntimeError("Can only fuse with generic bodies")

        rewriter.replace_op(user_op, streaming_region_op)
        rewriter.erase_op(op)


@dataclass(frozen=True)
class DartFuseOperationsPass(ModulePass):
    name = "dart-fuse-operations"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(FuseElementwisePattern()).rewrite_module(op)
