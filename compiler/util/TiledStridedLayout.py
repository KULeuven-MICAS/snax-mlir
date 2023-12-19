from typing import List, Optional
import numpy as np


class Stride:
    def __init__(self, stride: int, bound: int):
        assert stride > 0
        assert bound > 0
        self.stride = stride
        self.bound = bound
        self.level: Optional[int] = None
        self.dimension: Optional[int] = None

    def all_values(self) -> List[int]:
        return list(range(0, self.stride * self.bound, self.stride))

    def __str__(self) -> str:
        return f"{self.stride} x {self.bound}"


class TiledStride:
    def __init__(self, block: Stride, tile: Stride):
        self.block = block
        self.block.level = 1
        self.tile = tile
        self.tile.level = 0

    def get_stride(self, level) -> Stride:
        return self.tile if level == 0 else self.block

    def __str__(self) -> Stride:
        return f"[{self.tile.stride}, {self.block.stride}] x \
            [{self.tile.bound}, {self.block.bound}]"

    def __iter__(self) -> "TiledStride":
        self.iter_count = 0
        return self

    def __next__(self) -> Stride:
        if self.iter_count == 0:
            self.iter_count = 1
            return self.tile
        elif self.iter_count == 1:
            self.iter_count = 2
            return self.block
        else:
            raise StopIteration


class TiledStridedLayout:
    def __init__(self, strides: List[TiledStride]):
        self.dimensions = len(strides)
        self.strides = strides

        for i, stride in enumerate(self.strides):
            stride.tile.dimension = i
            stride.block.dimension = i

    def is_dense(self):
        """Check if there is a gap in the iteration space"""

        if self.self_overlaps():
            return False

        all_values = self.get_all_values()
        return np.max(all_values) == len(all_values) - 1

    def self_overlaps(self):
        """Duplicates"""
        u, c = np.unique(self.get_all_values(), return_counts=True)
        dup = u[c > 1]
        return len(dup) > 0

    def get_all_values(self):
        """Calculate all possible memory values of the strided layout"""

        result = np.array([0])

        for stride in self:
            # print(stride.all_values())
            next_stride = np.array(stride.all_values())
            result = np.squeeze(
                np.expand_dims(result, -1) + np.expand_dims(next_stride, 0)
            )

        result = result.flatten()
        return result

    def largest_common_contiguous_block(self, other: "TiledStridedLayout"):
        result: List[Stride] = []

        # does not work on illlegal layouts
        if self.self_overlaps():
            return result

        if other.self_overlaps():
            return result

        # find largest contiguous block, starting with stride = 1
        current_stride = 1
        while True:
            # find stride with correct current stride
            stride = next(
                (stride for stride in self if stride.stride == current_stride), None
            )
            if stride is None:
                return result

            assert stride.dimension < len(other.strides)
            compare_stride = other.strides[stride.dimension].get_stride(stride.level)

            if (
                stride.stride == compare_stride.stride
                and stride.bound == compare_stride.bound
            ):
                result.append(stride)
                current_stride = stride.stride * stride.bound
            else:
                return result

    def get_2d_dma_strategy(self, other: "TiledStridedLayout"):
        lcb = self.largest_common_contiguous_block(other)

        remaining_strides = [stride for stride in self if stride not in lcb]
        remaining_strides = sorted(remaining_strides, key=lambda x: x.bound)
        # TODO: check if bound is equal between self and other

        if len(remaining_strides) == 0:
            # TODO
            programmed_strides: List[Stride] = []
            pass
        elif len(remaining_strides) == 1:
            # TODO
            programmed_strides: List[Stride] = []
            pass
        else:
            dma_loop = remaining_strides[0]
            programmed_strides: List[Stride] = remaining_strides[
                1:
            ]  # TODO extend to more dimensions

        # for now just print the best strategy
        dma_loop_dst = other.strides[dma_loop.dimension].get_stride(dma_loop.level)

        print()
        pointer_strides = []
        for ps in programmed_strides:
            ps_dst = other.strides[ps.dimension].get_stride(ps.dimension)
            print(f"for(int i = 0; i < {ps.bound}; i++)")
            pointer_strides.append((ps_dst.stride, ps.stride))

        print(
            f'    snrt_dma_start_2d(*dst + \
                {" + ".join(["i*" + str(k[0]) for k in pointer_strides])}, \
            *src + {" + ".join(["i*" + str(k[0]) for k in pointer_strides])}, \
            size={lcb[-1].bound * lcb[-1].stride}, \
            dst_stride={dma_loop_dst.stride}, src_stride={dma_loop.stride}, \
            repeat={dma_loop.bound});'
        )
        print()

    def __str__(self):
        return "(" + ", ".join(map(str, self.strides)) + ")"

    def __iter__(self) -> "TiledStridedLayout":
        self.iter_idx = 0
        self.iter_sub = iter(self.strides[self.iter_idx])
        self.iter_nxt = next(self.iter_sub, None)
        return self

    def __next__(self) -> Stride:
        result = self.iter_nxt
        if result is None:
            raise StopIteration
        else:
            self.iter_nxt = next(self.iter_sub, None)
            if self.iter_nxt is None:
                self.iter_idx += 1
                if self.iter_idx < self.dimensions:
                    self.iter_sub = iter(self.strides[self.iter_idx])
                    self.iter_nxt = next(self.iter_sub, None)
            return result


if __name__ == "__main__":
    print("Hello There!")

    t1 = TiledStride(block=Stride(32, 2), tile=Stride(8, 4))
    t2 = TiledStride(block=Stride(4, 2), tile=Stride(1, 4))

    tsl1 = TiledStridedLayout([t1, t2])

    t3 = TiledStride(block=Stride(16, 2), tile=Stride(4, 4))
    t4 = TiledStride(block=Stride(32, 2), tile=Stride(1, 4))

    tsl2 = TiledStridedLayout([t3, t4])

    print(f"TSL1: {tsl1}")
    print(f"TSL2: {tsl2}")

    print(
        f"Largest Common Contiguous Block: {tsl1.largest_common_contiguous_block(tsl2)}"
    )
    print("   -> " + str(tsl1.largest_common_contiguous_block(tsl2)[0]))

    print("Tiling Strategy:")
    tsl1.get_2d_dma_strategy(tsl2)
