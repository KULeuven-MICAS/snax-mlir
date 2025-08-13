from abc import ABC

from snaxc.accelerators.streamers.streamers import (
    StreamerOpts,
)


class StreamerExtension(StreamerOpts, ABC):
    """
    Abstract base class for DMA extensions.
    This class defines the interface for DMA extensions that can be used
    with accelerators.
    """

    name: str
    csr_length: int

    def __eq__(self, value: object):
        if not isinstance(value, StreamerExtension):
            return False
        return self.name == value.name and self.csr_length == value.csr_length
