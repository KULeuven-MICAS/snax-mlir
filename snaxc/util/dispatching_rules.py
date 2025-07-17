from xdsl.dialects import linalg, memref
from xdsl.ir import Operation

from snaxc.dialects import dart


def dispatch_to_dm(op: Operation):
    """Rule to dispatch operations to the dm core:
    for now, this is only memref copy operations"""
    if isinstance(op, memref.CopyOp):
        return True
    if isinstance(op, dart.StreamingRegionOpBase):
        if isinstance(str_op := op.body.block.first_op, dart.GenericOp):
            kernel_op = str_op.body.block.first_op
            # Only dispatch to dm if the kernel is provided by a DMAExtension
            if any([isinstance(kernel_op, ext.supported_kernel.kernel_type) for ext in DMAExtension.__subclasses__()]):
                return True
    return False


def dispatch_to_compute(op: Operation):
    """
    Rule to dispatch operations to the dm core:
    for now, this is only linalg generic operations
    and streaming regions
    """
    if isinstance(op, linalg.GenericOp):
        return True
    if isinstance(op, dart.StreamingRegionOpBase):
        if isinstance(str_op := op.body.block.first_op, dart.GenericOp):
            kernel_op = str_op.body.block.first_op
            # Dont dispatch to compute if the kernel is provided by a DMAExtension
            if any([isinstance(kernel_op, ext.supported_kernel.kernel_type) for ext in DMAExtension.__subclasses__()]):
                return False
            return True
        return True

    return False
