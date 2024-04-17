from collections.abc import Iterable, Mapping

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator


class AcceleratorRegistry:
    """
    This registry is used to get specific accelerator conversions from
    a query based on an acc2.acceleratorop.
    """

    registered_accelerators = {"snax_hwpe_mult": SNAXHWPEMultAccelerator}

    def get_registry(self) -> Mapping[str, type[Accelerator]]:
        """
        Get a reference to an Accelerator interface based on a symbol name
        """
        return self.registered_accelerators

    def get_names(self) -> Iterable[str]:
        """
        Get an Iterable of strings that contains the names of the
        registered accelerators.
        """
        return self.registered_accelerators.keys()
