from xdsl.dialects import linalg, memref
from xdsl.ir import Operation

from snaxc.accelerators.acc_context import AccContext
from snaxc.accelerators.snax_xdma import SNAXXDMAAccelerator
from snaxc.accelerators.streamers.xdma_kernels import XDMA_KERNEL_SET
from snaxc.dialects import dart


def dispatch_to_dm(op: Operation, ctx: AccContext):
    """Rule to dispatch operations to the dm core:
    for now, this is only memref copy operations"""
    if isinstance(op, memref.CopyOp):
        return True
    if isinstance(op, dart.StreamingRegionOpBase):
        assert op.accelerator
        accelerator_type = ctx.get_acc(op.accelerator.data)
        if isinstance(accelerator_type, SNAXXDMAAccelerator) and isinstance(
            str_op := op.body.block.first_op, dart.GenericOp
        ):
            kernel_op = str_op.body.block.first_op
            # Only dispatch to dm if the kernel is provided by a StreamerExtension
            for xdma_kernel in XDMA_KERNEL_SET:
                if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                    return True
    return False


def dispatch_to_compute(op: Operation, ctx: AccContext):
    """
    Rule to dispatch operations to the dm core:
    for now, this is only linalg generic operations
    and streaming regions
    """
    if isinstance(op, linalg.GenericOp):
        return True
    if isinstance(op, dart.StreamingRegionOpBase):
        assert op.accelerator
        accelerator_type = ctx.get_acc(op.accelerator.data)
        if isinstance(accelerator_type, SNAXXDMAAccelerator) and isinstance(
            str_op := op.body.block.first_op, dart.GenericOp
        ):
            kernel_op = str_op.body.block.first_op
            # Dont dispatch to compute if the kernel is provided by a StreamerExtension
            for xdma_kernel in XDMA_KERNEL_SET:
                if xdma_kernel.supported_kernel.is_same_kernel(kernel_op):
                    # If the kernel is provided by a StreamerExtension, we do not dispatch to compute
                    return False
            return True
        return True

    return False
