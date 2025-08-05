from snaxc.accelerators.streamers.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasByteMask,
    HasChannelMask,
)

from .add_extension import AddExtension
from .maxpool_extension import MaxPoolExtension
from .memset_extension import MemSetExtension
from .rescale_down_extension import RescaleDownExtension
from .rescale_up_extension import RescaleUpExtension
from .streamer_extension import *
from .transpose_extension import TransposeExtension

XDMA_EXT_SET = (
    MaxPoolExtension,
    MemSetExtension,
    TransposeExtension,
    RescaleDownExtension,
    RescaleUpExtension,
    AddExtension,
)

STREAMER_OPT_MAP = {
    HasBroadcast().name: HasBroadcast,
    HasByteMask().name: HasByteMask,
    HasChannelMask().name: HasChannelMask,
    HasAddressRemap().name: HasAddressRemap,
    MaxPoolExtension().name: MaxPoolExtension,
    MemSetExtension().name: MemSetExtension,
    TransposeExtension().name: TransposeExtension,
    AddExtension().name: AddExtension,
    RescaleDownExtension().name: RescaleDownExtension,
    RescaleUpExtension().name: RescaleUpExtension,
}
