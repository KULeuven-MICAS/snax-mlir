from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class SoftMaxExtension(StreamerExtension):
    """
    Snax XDMA SoftMax Extension
    This extension is used to perform SoftMax operations on the XDMA core.
    """

    name = "softmax_ext"
    csr_length = 6

    def get_dma_extension_name(self) -> str:
        return self.name