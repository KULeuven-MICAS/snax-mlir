from .extensions import (
    AddExtension,
    MaxPoolExtension,
    MemSetExtension,
    RescaleDownExtension,
    RescaleUpExtension,
    TransposeExtension,
)
from .streamers import HasAddressRemap, HasBroadcast, HasByteMask, HasChannelMask

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
