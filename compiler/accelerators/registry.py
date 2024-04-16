from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator


def get_registered_accelerators() -> dict[str, type[Accelerator]]:
    return {"snax_hwpe_mult": SNAXHWPEMultAccelerator}
