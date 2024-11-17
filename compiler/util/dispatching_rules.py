from xdsl.dialects import linalg, memref

from compiler.dialects import stream


def dispatch_to_dm(op):
    """Rule to dispatch operations to the dm core:
    for now, this is only memref copy operations"""
    if isinstance(op, memref.CopyOp):
        return True
    if isinstance(op, linalg.Generic):
        if op.library_call is not None and op.library_call.data == "snax_xdma":
            return True
    return False


def dispatch_to_compute(op):
    """
    Rule to dispatch operations to the dm core:
    for now, this is only linalg generic operations
    and streaming regions
    """
    if isinstance(op, linalg.Generic):
        if op.library_call is not None and op.library_call.data in ("snax_xdma", "none"):
            return False
        return True
    if isinstance(op, stream.StreamingRegionOp):
        return True
    return False

def dispatch_alternative(op):
    if isinstance(op, linalg.Generic):
        if op.library_call is not None and op.library_call.data == "none":
            return True
