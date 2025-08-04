from dataclasses import dataclass
from math import ceil


@dataclass
class ConvLayer:
    name: str
    ox: int
    oy: int
    fx: int
    fy: int
    c: int
    k: int
    stride: int

    def tiling_sizes_oy(self) -> list[int]:
        """
        Returns a list of possible tile sizes for the output height (oy).
        """
        y_tiles: list[int] = []
        previous_tiled_oy = 0
        for tile_y in range(1, self.oy + 1):
            tiled_oy = ceil(self.oy / tile_y)
            if previous_tiled_oy == tiled_oy:
                continue
            y_tiles.append(tile_y)
            previous_tiled_oy = tiled_oy
        return y_tiles

    def tiling_sizes_k(self) -> list[int]:
        """
        Returns a list of possible tile sizes for the output channels (k).
        """
        k_tiles: list[int] = []
        previous_tiled_k = 0
        for tile_k in range(1, self.k + 1):
            tiled_k = ceil(self.k / tile_k)
            if previous_tiled_k == tiled_k:
                continue
            k_tiles.append(tile_k)
            previous_tiled_k = tiled_k
        return k_tiles

    @property
    def total_ops(self) -> int:
        return self.ox * self.oy * self.fx * self.fy * self.c * self.k


@dataclass
class ModelConfig:
    layers: list[ConvLayer]


@dataclass
class TiledConvLayer:
    layer: ConvLayer
    tile_k: int
    tile_y: int

    @property
    def tiled_ox(self) -> int:
        """
        Returns the tiled output width based on the layer's output width and the tiling size for output height.
        Ox is always padded to multiple of 8
        """
        if self.layer.ox % 8 == 0:
            return self.layer.ox
        else:
            return self.layer.ox + (8 - self.layer.ox % 8)

    @property
    def tiled_oy(self) -> int:
        """
        Returns the tiled output height based on the layer's output height and the tiling size for output height.
        """
        return ceil(self.layer.oy / self.tile_y)

    @property
    def tiled_k(self) -> int:
        """
        Returns the tiled output channels based on the layer's output channels and the tiling size for output channels.
        """
        return ceil(self.layer.k / self.tile_k)

    @property
    def tiled_ix(self) -> int:
        """
        Returns the tiled input width based on the layer's output width, filter width, stride, and dilation.
        """
        return (self.tiled_ox - 1) * self.layer.stride + (self.layer.fx - 1) + 1

    @property
    def tiled_iy(self) -> int:
        """
        Returns the tiled input height based on the layer's output height, filter height, stride, and dilation.
        """
        return (self.tiled_oy - 1) * self.layer.stride + (self.layer.fy - 1) + 1

    @property
    def padded_layer(self) -> ConvLayer:
        return ConvLayer(
            self.layer.name + "_padded",
            self.tiled_ox,
            self.tile_y * self.tiled_oy,
            self.layer.fx,
            self.layer.fy,
            self.layer.c,
            self.tile_k * self.tiled_k,
            self.layer.stride,
        )

    def input_tile_size(self) -> int:
        return self.tiled_ix * self.tiled_iy * self.layer.c

    def output_tile_size(self) -> int:
        return self.tiled_ox * self.tiled_oy * self.tiled_k

    def weight_tile_size(self) -> int:
        return self.layer.fx * self.layer.fy * self.layer.c * self.tiled_k

    def total_tile_size(self) -> int:
        return self.input_tile_size() + self.output_tile_size() + self.weight_tile_size()

    def iters(self) -> int:
        """
        Returns the number of iterations for the tiled operation
        """
        return self.tile_k * self.tile_y

    def total_transfer_size(self) -> int:
        """
        Returns the total transfer size for the tiled operation.
        """
        return self.total_tile_size() * self.iters()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TiledConvLayer):
            return False
        return (self.tile_k, self.tile_y) == (other.tile_k, other.tile_y)

    def __hash__(self) -> int:
        return hash((self.tile_k, self.tile_y))


@dataclass
class TiledConfig:
    layers: list[TiledConvLayer]
