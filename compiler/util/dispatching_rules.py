from xdsl.dialects import linalg, memref
from xdsl.ir import Operation

from compiler.dialects import dart


def dispatch_to_dm(op: Operation):
    """Rule to dispatch operations to the dm core:
    for now, this is only memref copy operations"""
    if isinstance(op, memref.CopyOp):
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
    if isinstance(op, dart.OperationOp):
        return True
    return False
