from .add_kernel import AddXDMA
from .avgpool_kernel import AvgPoolXDMA
from .maxpool_kernel import MaxPoolXDMA
from .rescale_kernel import RescaleDownXDMA, RescaleUpXDMA
from .softmax_kernel import SoftMaxXDMA
from .xdma_kernel import XDMAKernel  # pyright: ignore[reportUnusedImport]  # noqa: F401

XDMA_KERNEL_SET = (
    AddXDMA,
    MaxPoolXDMA,
    AvgPoolXDMA,
    RescaleDownXDMA,
    RescaleUpXDMA,
    SoftMaxXDMA,
)
