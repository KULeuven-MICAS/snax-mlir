from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class TransposeExtension(StreamerExtension):
    """
    Snax XDMA Transpose Extension
    This extension is used to perform transpose operations on the XDMA core.
    """

    name = "t"
    csr_length = 1
