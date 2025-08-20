from snaxc.accelerators.streamers.extensions.streamer_extension import StreamerExtension


class AddExtension(StreamerExtension):
    """
    Snax XDMA Add Extension
    This extension is used to perform Elementwise addition operations on the XDMA core.
    """

    name = "add_ext"
    csr_length = 1

    def get_dma_extension_name(self) -> str:
        return self.name
