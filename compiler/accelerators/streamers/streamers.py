from collections.abc import Sequence

from xdsl.utils.str_enum import StrEnum


class StreamerType(StrEnum):
    Reader = "r"
    Writer = "w"


class Streamer:
    """
    A representation of a single SNAX Streamer
    """

    type: StreamerType
    temporal_dim: int
    spatial_dim: int

    def __init__(self, type: StreamerType, temporal_dim: int, spatial_dim: int) -> None:
        self.type = type
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim


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

    def temporal_dim(self) -> int:
        """
        Return the temporal dimension of the streamers
        For now, assume all temporal dimensions are equal,
        so just take the first
        """
        return self.streamers[0].temporal_dim

    def spatial_dim(self) -> int:
        """
        Return the spatial dimension of the streamers
        For now, assume all spatial dimensions are equal,
        so just take the first
        """
        return self.streamers[0].spatial_dim
