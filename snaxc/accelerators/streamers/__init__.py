from .streamers import *
from .streamers import HasAddressRemap, HasBroadcast, HasByteMask, HasChannelMask
from .xdma_extensions import AddExtension, MaxPoolExtension, MemSetExtension, TransposeExtension

STREAMER_OPT_MAP = {
HasBroadcast().name: HasBroadcast,
HasByteMask().name: HasByteMask,
HasChannelMask().name: HasChannelMask,
HasAddressRemap().name: HasAddressRemap,
MaxPoolExtension().name: MaxPoolExtension,
MemSetExtension().name: MemSetExtension,
TransposeExtension().name: TransposeExtension,
AddExtension().name: AddExtension,
}
