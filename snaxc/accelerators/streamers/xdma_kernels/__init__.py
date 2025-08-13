from .add_kernel import AddXDMA
from .avgpool_kernel import AvgPoolXDMA
from .maxpool_kernel import MaxPoolXDMA
from .rescale_kernel import RescaleDownXDMA, RescaleUpXDMA
from .xdma_kernel import XDMAKernel

XDMA_KERNEL_SET = (
    AddXDMA,
    AvgPoolXDMA,
    MaxPoolXDMA,
    RescaleDownXDMA,
    RescaleUpXDMA,
)
