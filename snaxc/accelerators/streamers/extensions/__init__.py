from snaxc.accelerators.streamers.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasByteMask,
    HasChannelMask,
)

from .add_extension import AddExtension
from .add_to_long_extension import AddToLongExtension
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
    AddToLongExtension().name: AddToLongExtension,
    MemSetExtension().name: MemSetExtension,
    TransposeExtension().name: TransposeExtension,
    AddExtension().name: AddExtension,
    RescaleDownExtension().name: RescaleDownExtension,
    RescaleUpExtension().name: RescaleUpExtension,
}
