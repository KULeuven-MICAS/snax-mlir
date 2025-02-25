from collections.abc import Iterable

from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.dialects.builtin import ShapedType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.dispatching import DispatchTemplate
from snaxc.accelerators.snax import SNAXStreamer
from snaxc.dialects.accfg import AcceleratorOp
from snaxc.dialects.kernel import KernelOp


class DispatchTemplatePattern(RewritePattern):
    """
    Dispatch kernels to accelerators based on their specified
    compute kernel template.
    """

    accelerators: Iterable[type[DispatchTemplate]] = []

    def __init__(self, accelerators: Iterable[type[DispatchTemplate]]) -> None:
        self.accelerators = accelerators

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # check if already dispatched
        if linalg_op.library_call:
            return

        # extract kernel ops from linalg body
        linalg_body_ops = iter(linalg_op.body.block.ops)

        kernel_op = next(linalg_body_ops)
        if not isinstance(kernel_op, KernelOp):
            return

        # expect linalg yield next
        if not isinstance(next(linalg_body_ops), linalg.YieldOp):
            return

        matched_accelerator: type[DispatchTemplate] | None = None

        for accelerator in self.accelerators:
            if matched_accelerator:
                # already match found
                break

            for supported_kernel in accelerator.supported_kernels:
                # check if kernel supported
                if supported_kernel.kernel_type is not type(kernel_op):
                    # no, continue
                    continue

                # kernel supported, check operand types
                for template_el_type, kernel_el in zip(
                    supported_kernel.operand_types,
                    (*kernel_op.operands, *kernel_op.results),
                    strict=True,
                ):
                    if template_el_type != kernel_el.type:
                        # no match, continue
                        continue

                # kernel supported & operand types matched successfully
                matched_accelerator = accelerator
                break

        if not matched_accelerator:
            return

        # set linalg op library call
        library_call = matched_accelerator.name

        # optional streaming extension for custom operands:
        if issubclass(matched_accelerator, SNAXStreamer):
            suffix = "_stream"
            # check if no dynamic operands
            for operand in (
                o.type for o in linalg_op.operands if isinstance(o.type, ShapedType)
            ):
                if -1 in operand.get_shape():
                    suffix = ""
                    break

            library_call = library_call + suffix

        linalg_op.library_call = builtin.StringAttr(library_call)


class DispatchKernels(ModulePass):
    name = "dispatch-kernels"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # find all accelerator ops in the IR
        assert isinstance(ctx, AccContext)
        accelerators: list[type[DispatchTemplate]] = []
        for accelerator_op in op.ops:
            if isinstance(accelerator_op, AcceleratorOp):
                accelerator_type = ctx.get_acc(
                    str(accelerator_op.properties["name"])[1:]
                )
                if issubclass(accelerator_type, DispatchTemplate):
                    accelerators.append(accelerator_type)

        # dispatch
        PatternRewriteWalker(DispatchTemplatePattern(accelerators)).rewrite_module(op)
