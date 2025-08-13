from snaxc.accelerators.streamers.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasByteMask,
    HasChannelMask,
)

from .add_extension import AddExtension
from .avgpool_extension import AvgPoolExtension
from .maxpool_extension import MaxPoolExtension
from .memset_extension import MemSetExtension
from .rescale_extension import RescaleDownExtension, RescaleUpExtension
from .streamer_extension import *
from .transpose_extension import TransposeExtension

STREAMER_OPT_MAP = {
    HasBroadcast().name: HasBroadcast,
    HasByteMask().name: HasByteMask,
    HasChannelMask().name: HasChannelMask,
    HasAddressRemap().name: HasAddressRemap,
    MaxPoolExtension().name: MaxPoolExtension,
    AvgPoolExtension().name: AvgPoolExtension,
    MemSetExtension().name: MemSetExtension,
    TransposeExtension().name: TransposeExtension,
    AddExtension().name: AddExtension,
    RescaleDownExtension().name: RescaleDownExtension,
    RescaleUpExtension().name: RescaleUpExtension,
}
