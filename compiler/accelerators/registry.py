from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator


def get_registered_accelerators() -> dict[str, type[Accelerator]]:
    """
    Get a reference to an Accelerator interface based on a symbol name
    This registry is used to get specific accelerator conversions from
    a query based on an acc2.acceleratorop.
    """
    return {"snax_hwpe_mult": SNAXHWPEMultAccelerator}
