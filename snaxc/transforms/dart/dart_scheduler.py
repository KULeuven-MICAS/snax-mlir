from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.dialects import dart
from snaxc.ir.dart.access_pattern import Schedule, SchedulePattern
from snaxc.ir.dart.scheduler import scheduler


@dataclass
class AutoflowScheduler(RewritePattern):
    """
    A pass to convert streaming region operations to schedules.

    Here, the operation is scheduled to an accelerator according to the accelerator template.
    """

    ctx: AccContext

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dart.OperationOp, rewriter: PatternRewriter):
        assert op.accelerator
        accelerator_type = self.ctx.get_acc(op.accelerator.data)
        assert isinstance(accelerator_type, SNAXStreamer)
        template = accelerator_type.get_template(op)

        # Make sure the operands are memrefs
        for memref_operand in op.operands:
            if not isinstance(memref_operand.type, builtin.MemRefType):
                return

        # First, run the stream scheduling algorithm
        schedule_bounds = tuple(op.get_static_pattern_bounds())
        schedule = Schedule(
            SchedulePattern(schedule_bounds, pattern.data)
            for pattern in op.patterns.data
        )
        schedule = scheduler(template, schedule)

        schedule_op = dart.ScheduleOp(
            op.inputs,
            op.outputs,
            ArrayAttr([AffineMapAttr(s.pattern.to_affine_map()) for s in schedule]),
            rewriter.move_region_contents_to_new_regions(op.body),
            schedule[0].bounds,
            [[]],
            op.accelerator,
            op.result_types,
        )

        rewriter.replace_matched_op(schedule_op)


@dataclass(frozen=True)
class DartSchedulerPass(ModulePass):
    name = "dart-scheduler"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        assert isinstance(ctx, AccContext)
        PatternRewriteWalker(AutoflowScheduler(ctx)).rewrite_module(op)
