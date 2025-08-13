from .add_kernel import AddXDMA
from .maxpool_kernel import MaxPoolXDMA
from .rescale_kernel import RescaleDownXDMA, RescaleUpXDMA
from .xdma_kernel import XDMAKernel  # pyright: ignore[reportUnusedImport]  # noqa: F401

XDMA_KERNEL_SET = (
    AddXDMA,
    MaxPoolXDMA,
    RescaleDownXDMA,
    RescaleUpXDMA,
)
