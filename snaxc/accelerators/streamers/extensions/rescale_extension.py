from abc import ABC

from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class RescaleExtension(StreamerExtension, ABC):
    """
    Snax XDMA Rescale Extension
    This extension is used to perform Rescale operations on the XDMA core.
    """

    name: str
    csr_length = 4


class RescaleDownExtension(RescaleExtension):
    """
    Snax XDMA Rescale Down Extension
    """

    name = "rescale_down_ext"


class RescaleUpExtension(RescaleExtension):
    """
    Snax XDMA Rescale Up Extension
    """

    name = "rescale_up_ext"
