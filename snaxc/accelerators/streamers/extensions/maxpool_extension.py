from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class MaxPoolExtension(StreamerExtension):
    """
    Snax XDMA MaxPool Extension
    This extension is used to perform max pooling operations on the XDMA core.
    """

    name = "maxpool_ext"
    csr_length = 1
