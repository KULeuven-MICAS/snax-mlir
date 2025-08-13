from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class AvgPoolExtension(StreamerExtension):
    """
    Snax XDMA AvgPool Extension
    This extension is used to perform average pooling operations on the XDMA core.
    """

    name = "avgpool_ext"
    csr_length = 1
