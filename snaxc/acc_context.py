from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from xdsl.context import Context

from snaxc.accelerators import Accelerator


@dataclass
class AccContext(Context):
    """
    Context that additionally allows to register and query accelerators
    """

    _registered_accelerators: dict[str, Callable[[], type[Accelerator]]] = field(
        default_factory=dict
    )

    def clone(self) -> "Context":
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

    def get_accelerator(self, name: str) -> type[Accelerator]:
        """
        Get an operation class from its name if it exists.
        If the accelerator is not registered, raise an exception.
        """
        if (accelerator := self.get_optional_accelerator(name)) is None:
            raise Exception(f"Accelerator {name} is not registered")
        return accelerator

    @property
    def registered_accelerator_names(self) -> Iterable[str]:
        """
        Returns the names of all registered accelerators. Not valid across mutations of this object.
        """
        return self._registered_dialects.keys()
