from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class AddToLongExtension(StreamerExtension):
    """
    Snax XDMA AddToLong Extension
    This extension is used to perform add to long operations on the XDMA core
    where the output width is larger than the input width.
    """

    name = "add_to_long_ext"
    csr_length = 1
