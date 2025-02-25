from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TypeVar

from xdsl.context import Context
from xdsl.ir import Operation
from xdsl.parser import ModuleOp
from xdsl.traits import SymbolTable

from snaxc.accelerators.accelerator import Accelerator
from snaxc.dialects.accfg import AcceleratorOp

T = TypeVar("T")


@dataclass
class AccContext(Context):
    """
    Context that additionally allows to register and query accelerators
    """

    _registered_accelerators: dict[str, Callable[[], type[Accelerator]]] = field(
        default_factory=dict
    )

    def clone(self) -> "AccContext":
        return AccContext(
            self.allow_unregistered,
            self._loaded_dialects.copy(),
            self._loaded_ops.copy(),
            self._loaded_attrs.copy(),
            self._registered_dialects.copy(),
            self._registered_accelerators.copy(),
        )

    def register_accelerator(
        self, name: str, accelerator_factory: Callable[[], type[Accelerator]]
    ) -> None:
        """
        Register an accelerator without loading it.
        The accelerator is only loaded in the context
        if the name is requested from the AccContext
        """
        if name in self._registered_accelerators:
            raise ValueError(f"'{name}' accelerator is already registered")
        self._registered_accelerators[name] = accelerator_factory

    def get_optional_accelerator(self, name: str) -> type[Accelerator] | None:
        """
        Get an operation class from its name if it exists.
        If the accelerator is not registered, raise an exception.
        """
        if name in self._registered_accelerators:
            return self._registered_accelerators[name]()
        return None

    def get_acc(self, name: str) -> type[Accelerator]:
        """
        Get an operation class from its name if it exists.
        If the accelerator is not registered, raise an exception.
        """
        if (accelerator := self.get_optional_accelerator(name)) is None:
            raise Exception(f"Accelerator {name} is not registered")
        return accelerator

    def get_acc_op_from_module(
        self, name: str, module: ModuleOp
    ) -> tuple[AcceleratorOp, type[Accelerator]]:
        """
        Perform a symbol table lookup for the accelerator op in the IR
        and then get the corresponding the Accelerator interface from
        the accelerator registry.
        Returns both the looked up accelerator op and the Accelerator interface
        """
        acc_op = find_accelerator_op(module, name)
        if acc_op is None:
            raise Exception(
                f"Symbol Table lookup failed for accelerator '{name}'. "
                "Is the symbol declared by an accfg.accelerator op in the module?"
            )
        return acc_op, self.get_acc(acc_op.name_prop.string_value())

    @property
    def registered_accelerator_names(self) -> Iterable[str]:
        """
        Returns the names of all registered accelerators. Not valid across mutations of this object.
        """
        return self._registered_accelerators.keys()


def find_accelerator_op(op: Operation, accelerator_str: str) -> AcceleratorOp | None:
    """
    Finds the accelerator op with a given symbol name in the ModuleOp of
    a given operation. Returns None if not found.
    """

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
