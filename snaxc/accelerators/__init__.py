from collections.abc import Callable

from snaxc.accelerators.acc_context import *
from snaxc.accelerators.accelerator import Accelerator


def get_all_accelerators() -> dict[str, Callable[[], Accelerator]]:
    """Return the list of all available passes."""

    def get_snax_alu():
        from snaxc.accelerators.snax_alu import SNAXAluAccelerator

        return SNAXAluAccelerator()

    def get_snax_gemm():
        from snaxc.accelerators.snax_gemm import SNAXGEMMAccelerator

        return SNAXGEMMAccelerator()

    def get_snax_gemmx():
        from snaxc.accelerators.snax_gemmx import SNAXGEMMXAccelerator

        return SNAXGEMMXAccelerator()

    def get_snax_hwpe_mult():
        from snaxc.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator

        return SNAXHWPEMultAccelerator()

    def get_gemmini():
        from snaxc.accelerators.gemmini import GemminiAccelerator

        return GemminiAccelerator()

    return {
        "snax_alu": get_snax_alu,
        "snax_gemm": get_snax_gemm,
        "snax_gemmx": get_snax_gemmx,
        "snax_hwpe_mult": get_snax_hwpe_mult,
        "gemmini": get_gemmini,
    }
