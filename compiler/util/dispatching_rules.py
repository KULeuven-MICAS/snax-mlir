from xdsl.dialects import linalg, memref


def dispatch_to_dm(op):
    """Rule to dispatch operations to the dm core:
    for now, this is only memref copy operations"""
    if isinstance(op, memref.CopyOp):
        return True
    return False


def dispatch_to_compute(op):
    """Rule to dispatch operations to the dm core:
    for now, this is only linalg generic operations"""
    if isinstance(op, linalg.Generic):
        return True
    return False
