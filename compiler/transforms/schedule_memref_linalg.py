from collections.abc import Sequence

from xdsl.context import MLContext
from xdsl.dialects import builtin, memref_stream
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.util.canonicalize_affine import canonicalize_map


class ScheduleMemrefLinalgRewriter(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref_stream.GenericOp, rewriter: PatternRewriter):

        if not (isinstance(op.library_call, builtin.StringAttr) and op.library_call.data == "snax_alu"):
            return

        # only handle snax_alu for now
        template_alu_schedule = AffineMap.from_callable(lambda x, y: (4 * x + y,))
        template_alu_bounds = (None, 4)

        # create new patterns, compatible with accelerator
        new_patterns: Sequence[builtin.AffineMapAttr] = []

        for pattern in op.indexing_maps:
            # TODO: make for loop work for dim in range(template_alu_schedule.num_dims):
            dim = 1

            # extract last n dims from template and pattern
            template_check = canonicalize_map(
                template_alu_schedule.replace_dims_and_symbols(
                    [AffineConstantExpr(0) for _ in range(template_alu_schedule.num_dims - dim)]
                    + [AffineDimExpr(i) for i in range(dim)],
                    [],
                    dim,
                    0,
                )
            )

            pattern_check = canonicalize_map(
                pattern.data.replace_dims_and_symbols(
                    [AffineConstantExpr(0) for _ in range(pattern.data.num_dims - dim)]
                    + [AffineDimExpr(i) for i in range(dim)],
                    [],
                    dim,
                    0,
                )
            )

            # check for pattern compatibility
            if template_check != pattern_check:
                # not a compatible stream pattern, panic!
                raise RuntimeError("incompatible memref stream detected")

            # check for bounds compatiblity
            if template_alu_bounds[-dim]:
                # bound detected, check if it is larger than the pattern bound
                tb = template_alu_bounds[-dim]
                assert isinstance(tb, int)
                if op.bounds.data[-dim].value.data is None or op.bounds.data[-dim].value.data > tb:
                    # bound of operation exceeds bound of template, create transformation for pattern
                    transform_map = AffineMap.from_callable(lambda x, y: (template_alu_bounds[-1] * x + y,))
                    new_pattern = pattern.data.compose(transform_map)

                    # transform bounds
                    new_bounds = op.bounds.data
                    transform_bounds_map = AffineMap(
                        1,
                        0,
                        (
                            (AffineDimExpr(0) - 1).floor_div(template_alu_bounds[-1]) + 1,
                            ((AffineDimExpr(0) - 1) % template_alu_bounds[-1]) + 1,
                        ),
                    )
                    new_bounds = transform_bounds_map.eval([op.bounds.data[0].value.data], [])

                    # create new stride pattern
                    new_patterns.append(builtin.AffineMapAttr(new_pattern))

                    new_bounds = builtin.ArrayAttr([builtin.IntegerAttr(b, builtin.IndexType()) for b in new_bounds])
                else:
                    new_patterns.append(pattern)
                    new_bounds = op.bounds
            else:
                new_patterns.append(pattern)
                new_bounds = op.bounds


        # TODO: fix this
        iterator_types = builtin.ArrayAttr(op.iterator_types.data * 2)

        new_op = memref_stream.GenericOp(
            inputs=op.inputs,
            outputs=op.outputs,
            inits=op.inits,
            body=rewriter.move_region_contents_to_new_regions(op.body),
            indexing_maps=builtin.ArrayAttr(new_patterns),
            iterator_types=iterator_types,
            bounds=new_bounds,
            init_indices=op.init_indices,
            doc=op.doc,
            library_call=op.library_call,
        )

        rewriter.replace_matched_op(new_op)


class ScheduleMemrefLinalg(ModulePass):
    """
    A pass to schedule a memref stream according
    to an accelerator definition.
    """

    name = "schedule-memref-linalg"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp):
        PatternRewriteWalker(ScheduleMemrefLinalgRewriter(), apply_recursively=False).rewrite_module(op)
