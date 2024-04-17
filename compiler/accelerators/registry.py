from collections.abc import Iterable

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator
from compiler.dialects.acc import AcceleratorOp


class AcceleratorRegistry:
    """
    This registry is used to get specific accelerator conversions from
    a query based on an acc2.acceleratorop.
    """

    registered_accelerators = {"snax_hwpe_mult": SNAXHWPEMultAccelerator}

    def get_acc_info(self, acc_op: AcceleratorOp) -> type[Accelerator]:
        """
        Get a reference to an Accelerator interface based on a symbol name
        If the requested symbol name is not available, throw a RuntimeError
        """
        acc_name = acc_op.name_prop.string_value()
        try:
            acc_info = self.registered_accelerators[acc_name]
        except KeyError:
            raise RuntimeError(
                f"'{acc_name}' is not a registered accelerator."
                f"Registered accelerators: {','.join(self.get_names())}"
            )
        return acc_info

    def get_names(self) -> Iterable[str]:
        """
        Get an Iterable of strings that contains the names of the
        registered accelerators.
        """
        return self.registered_accelerators.keys()
