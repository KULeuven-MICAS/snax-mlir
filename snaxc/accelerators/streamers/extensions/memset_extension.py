from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class MemSetExtension(StreamerExtension):
    """
    Snax XDMA MemSet Extension
    This extension is used to perform memory set operations on the XDMA core.
    """

    name = "memset_ext"
    csr_length = 1
