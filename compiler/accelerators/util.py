from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Operation
from xdsl.traits import SymbolTable

from compiler.dialects.accfg import AcceleratorOp


def find_accelerator_op(
    op: Operation, accelerator_str: str | StringAttr
) -> AcceleratorOp | None:
    """
    Finds the accelerator op with a given symbol name in the ModuleOp of
    a given operation. Returns None if it is not found.
    """

    # make accelerator_str a str
    if isinstance(accelerator_str, StringAttr):
        accelerator_str = accelerator_str.data

    # find the module op
    module_op = op
    while module_op and not isinstance(module_op, ModuleOp):
        module_op = module_op.parent_op()
    if not module_op:
        raise RuntimeError("Module Op not found")

    trait = module_op.get_trait(SymbolTable)
    assert trait is not None
    acc_op = trait.lookup_symbol(module_op, accelerator_str)

    assert isinstance(acc_op, AcceleratorOp | None)

    return acc_op
