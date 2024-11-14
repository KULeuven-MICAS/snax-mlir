from collections.abc import Iterable, Sequence
from typing import Literal

from typing_extensions import deprecated
from xdsl.utils.str_enum import StrEnum


class StreamerType(StrEnum):
    # Streamer with read capabilities
    Reader = "r"
    # Streamer with write capabilities
    Writer = "w"


class StreamerOpts(StrEnum):
    # Streamer with transpose capabilities
    HasTranspose = "t"
    # Streamer with channel mask capabilities
    HasChannelMask = "c"
    # Weird address remap thingy
    HasAddressRemap = "r"
    # Broadcasting
    HasBroadcast = "b"



class StreamerFlag(StrEnum):
    """
    Enum that specifies special flags for a streamer dimension.

    Attributes:
    -----------
    - Normal: 'n'
      Indicates that no special flags apply.
    - Irellevant : 'i'
      Indicates a dimension is irrelevant for this operand.
      In all cases, a zero should be set to the stride of this dimension.
      For spatial dims, the resulting value is assigned to a "virtual streamer"
      and will not be programmed.
    - Reuse : 'r'
      Only valid for temporal dims. Indicates that the values
      will be reused temporally and should only be fetched once.
      This results in the bound values being for this dimension fixed to 1.
    """

    Normal = "n"
    Irrelevant = "i"
    Reuse = "r"

    def __bool__(self) -> bool:
        """
        Overrides the default boolean conversion to ensure that the 'Normal'
        flag evaluates to False and all other flags evaluate to true
        """
        return self is not StreamerFlag.Normal


class Streamer:
    """
    A representation of a single SNAX Streamer
    """

    type: StreamerType
    temporal_dims: tuple[StreamerFlag, ...]
    spatial_dims: tuple[StreamerFlag, ...]

    opts: set[StreamerOpts]

    def __init__(
        self,
        type: StreamerType,
        temporal_dims: Sequence[StreamerFlag | Literal["n", "i", "r"]],
        spatial_dims: Sequence[StreamerFlag | Literal["n", "i", "r"]],
        opts: Iterable[StreamerOpts] = [],
    ) -> None:
        self.type = type
        temporal_dims = [
            f if isinstance(f, StreamerFlag) else StreamerFlag(f) for f in temporal_dims
        ]
        self.temporal_dims = tuple(temporal_dims)
        spatial_dims = [
            f if isinstance(f, StreamerFlag) else StreamerFlag(f) for f in spatial_dims
        ]
        self.spatial_dims = tuple(spatial_dims)
        self.opts = set(opts)

    @classmethod
    def from_dim(
        cls, type: StreamerType, temporal_dim: int, spatial_dim: int
    ) -> "Streamer":
        """
        Returns a streamer with a specified temporal dim and spatial dim
        as integers, without any flags set.
        """
        return cls(
            type,
            (StreamerFlag.Normal,) * temporal_dim,
            (StreamerFlag.Normal,) * spatial_dim,
        )

    @property
    def temporal_dim(self):
        return len(self.temporal_dims)

    @property
    def spatial_dim(self):
        return len(self.spatial_dims)


class StreamerConfiguration:
    """
    A representation for a SNAX Streamer Configuration.
    The configuration consists of one of more Streamer objects,
    one for each operand of the accelerator.
    """

    streamers: Sequence[Streamer]

    def __init__(self, streamers: Sequence[Streamer]):
        assert len(streamers)
        self.streamers = streamers

    def size(self) -> int:
        """
        Return the number of streamers in the configuration
        """
        return len(self.streamers)

    @deprecated("Please do not use this function, it is only valid in trivial cases")
    def temporal_dim(self) -> int:
        """
        Return the temporal dimension of the streamers
        For now, assume all temporal dimensions are equal,
        so just take the first
        """
        return self.streamers[0].temporal_dim

    @deprecated("Please do not use this function, it is only valid in trivial cases")
    def spatial_dim(self) -> int:
        """
        Return the spatial dimension of the streamers
        For now, assume all spatial dimensions are equal,
        so just take the first
        """
        return self.streamers[0].spatial_dim
