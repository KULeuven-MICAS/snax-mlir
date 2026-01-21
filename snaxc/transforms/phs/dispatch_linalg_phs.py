from collections.abc import Iterable

from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.dialects.builtin import DYNAMIC_INDEX, ShapedType
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from snaxc.accelerators import AccContext
from snaxc.accelerators.snax_phs import SNAXPHSAccelerator
from snaxc.dialects.accfg import AcceleratorOp
from snaxc.phs.decode import MappingNotFoundError, decode_abstract_graph
from snaxc.phs.encode import convert_generic_body_to_phs


class DispatchLinalgPhsPattern(RewritePattern):
    """
    Dispatch kernels to accelerators based on their specified
    compute kernel template.
    """

    accelerators: Iterable[SNAXPHSAccelerator] = []

    def __init__(self, accelerators: Iterable[SNAXPHSAccelerator]) -> None:
        self.accelerators = accelerators

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # if already dispatched, don't dispatch again
        if linalg_op.library_call:
            return

        to_map_pe = convert_generic_body_to_phs(linalg_op, "candidate", rewriter)
        for accelerator in self.accelerators:
            try:
                # Don't use the values, just see if it works
                decode_abstract_graph(accelerator.pe, to_map_pe)
                # set linalg op library call
                library_call = accelerator.name

                # optional streaming extension for custom operands:
                suffix = "_stream"
                # check if no dynamic operands
                for operand in (o.type for o in linalg_op.operands if isinstance(o.type, ShapedType)):
                    if DYNAMIC_INDEX in operand.get_shape():
                        suffix = ""
                        break

                library_call = library_call + suffix
                linalg_op.library_call = builtin.StringAttr(library_call)
                break
            except MappingNotFoundError:
                continue


class DispatchLinalgPHS(ModulePass):
    name = "dispatch-linalg-phs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # find all accelerator ops in the IR
        assert isinstance(ctx, AccContext)
        accelerators: list[SNAXPHSAccelerator] = []
        for accelerator_op in op.ops:
            if isinstance(accelerator_op, AcceleratorOp):
                accelerator_type = ctx.get_acc(accelerator_op.get_acc_name())
                if isinstance(accelerator_type, SNAXPHSAccelerator):
                    accelerators.append(accelerator_type)

        # dispatch
        PatternRewriteWalker(DispatchLinalgPhsPattern(accelerators)).rewrite_module(op)
