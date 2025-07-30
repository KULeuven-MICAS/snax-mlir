from .add_extension import AddExtension
from .maxpool_extension import MaxPoolExtension
from .memset_extension import MemSetExtension
from .rescale_down_extension import RescaleDownExtension
from .streamer_extension import *
from .transpose_extension import TransposeExtension

XDMA_EXT_SET = (
    MaxPoolExtension,
    MemSetExtension,
    TransposeExtension,
    RescaleDownExtension,
    AddExtension,
)
