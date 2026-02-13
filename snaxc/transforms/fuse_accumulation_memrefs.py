"""Transform pass that fuses accumulation memrefs in dart.operation.

After snax-bufferize, a dart.operation for d = b*c + d may have separate
memrefs for the input accumulator (d_in) and the output buffer (d_out).
The output is typically a fresh memref.alloc() while the input contains
the initial accumulation values. The body contains an explicit Add
GenericOp that adds the QMac result to d_in.

This pass identifies this pattern and fuses them by:
  1. Replacing all uses of the alloc (d_out) with d_in everywhere,
     including in func.return.
  2. Removing the explicit kernel.add GenericOp from the body, since
     the GEMMX hardware QMac inherently performs the accumulation
     via its C streamer.
  3. Dropping the accumulation-input block arg and its pattern,
     reducing the operation from 4 to 3 operands/patterns/block-args.

Before:
  %d_in  = memref.get_global @... : memref<MxNxT>
  %d_out = memref.alloc() : memref<MxNxT>
  "dart.operation"(%a, %b, %d_in, %d_out)
      <{operandSegmentSizes = array<i32: 3, 1>,
        patterns = [map_a, map_b, map_d, map_d]}>
  ({^bb0(%s_a, %s_b, %s_d_in, %s_d_out):
      %q = dart.generic(%s_a, %s_b, 0, 0) { kernel.qmac }
      %a = dart.generic(%q, %s_d_in) { kernel.add }
      dart.yield %a
  })
  func.return %d_out, ...

After:
  %d = memref.get_global @... : memref<MxNxT>
  "dart.operation"(%a, %b, %d)
      <{operandSegmentSizes = array<i32: 2, 1>,
        patterns = [map_a, map_b, map_d]}>
  ({^bb0(%s_a, %s_b, %s_d):
      %q = dart.generic(%s_a, %s_b, 0, 0) { kernel.qmac }
      dart.yield %q
  })
  func.return %d, ...
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, memref
from xdsl.dialects.builtin import UnitAttr
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter

from snaxc.dialects import dart, kernel
from snaxc.dialects.dart import OperationOp


def _try_fuse(op: OperationOp) -> bool:
    """Try to fuse accumulation memrefs on a single dart.operation.

    Returns True if the operation was transformed.
    """
    # We need at least one input and exactly one output
    if len(op.outputs) != 1 or len(op.inputs) < 1:
        return False

    output = op.outputs[0]
    accum_input = op.inputs[-1]

    # Both must have the same memref type
    if output.type != accum_input.type:
        return False

    # Output must come from a memref.alloc
    if not isinstance(output.owner, memref.AllocOp):
        return False

    # The last two affine-map patterns must match
    patterns = op.patterns.data
    if len(patterns) < 2 or patterns[-2] != patterns[-1]:
        return False

    # Check body structure: expect QMac GenericOp followed by Add GenericOp
    block = op.body.block
    qmac_generic = None
    add_generic = None
    for body_op in block.ops:
        if isinstance(body_op, dart.GenericOp):
            inner_first = body_op.body.block.first_op
            if isinstance(inner_first, (kernel.QMacOp, kernel.MacOp)):
                qmac_generic = body_op
            elif isinstance(inner_first, kernel.AddOp):
                add_generic = body_op

    if qmac_generic is None or add_generic is None:
        return False

    # ---- All checks passed: fuse ----
    alloc_op = output.owner
    n_inputs = len(op.inputs)

    # Capture new operands/patterns before modifying
    new_inputs = list(op.inputs[:-1])       # [A, B]  (drop D_in)
    new_outputs = [accum_input]              # [D = accum_input]
    new_patterns = builtin.ArrayAttr(list(patterns[:-1]))  # drop duplicate

    # Step 1: Replace all uses of the alloc result with accum_input.
    # This fixes func.return and any other consumers.
    alloc_op.results[0].replace_by(accum_input)
    Rewriter.erase_op(alloc_op)

    # Step 2: Modify the body in-place
    # Replace Add result with QMac result, then erase Add
    add_generic.results[0].replace_by(qmac_generic.results[0])
    add_generic.detach()
    add_generic.erase()

    # Mark the QMac generic so downstream passes know accumulation is needed
    qmac_generic.accumulates = UnitAttr()

    # Erase the accumulation-input block arg (s_d_in at index n_inputs-1)
    accum_block_arg = block.args[n_inputs - 1]
    block.erase_arg(accum_block_arg)

    # Step 3: Move body and reconstruct the OperationOp
    new_body = Rewriter.move_region_contents_to_new_regions(op.body)

    new_op = OperationOp(
        inputs=new_inputs,
        outputs=new_outputs,
        patterns=new_patterns,
        body=new_body,
        accelerator=op.accelerator,
    )

    Rewriter.replace_op(op, [new_op])
    return True


@dataclass(frozen=True)
class FuseAccumulationMemrefsPass(ModulePass):
    """Pass that fuses accumulation memrefs in dart.operation ops.

    Should be run after snax-bufferize to remove the redundant alloc and
    explicit Add, letting the hardware handle accumulation directly.
    """

    name = "fuse-accumulation-memrefs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for dart_op in list(op.walk()):
            if isinstance(dart_op, OperationOp):
                _try_fuse(dart_op)
