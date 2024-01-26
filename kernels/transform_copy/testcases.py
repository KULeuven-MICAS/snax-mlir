from compiler.ir.tsl.stride import Stride
from compiler.ir.tsl.tiled_stride import TiledStride
from compiler.ir.tsl.tiled_strided_layout import TiledStridedLayout

testcases = [
    {
        "name": "equal_layout_dynamic_lcb_4",
        "array_sizes": [64, 256, 1024],
        "shape": lambda _: [-1, -1],
        "tsldst": lambda _: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(16, None),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(None, None), Stride(None, 4)]),
            ]
        ),
        "tslsrc": lambda _: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(16, None),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(None, None), Stride(None, 4)]),
            ]
        ),
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "equal_layout_dynamic_lcb_8",
        "array_sizes": [256, 1024],
        "shape": lambda _: [-1, -1],
        "tsldst": lambda _: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(32, None),
                        Stride(4, 8),
                    ]
                ),
                TiledStride([Stride(None, None), Stride(None, 4)]),
            ]
        ),
        "tslsrc": lambda _: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(32, None),
                        Stride(4, 8),
                    ]
                ),
                TiledStride([Stride(None, None), Stride(None, 4)]),
            ]
        ),
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "equal_layout_static",
        "array_sizes": [256, 1024],
        "shape": lambda size: [size, size],
        "tsldst": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(16, size // 4),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(16 * size, size // 4), Stride(16 * size // 4, 4)]),
            ]
        ),
        "tslsrc": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(16, size // 4),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(16 * size, size // 4), Stride(16 * size // 4, 4)]),
            ]
        ),
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "transform_block_transpose",
        "array_sizes": [64, 256, 1024],
        "shape": lambda size: [size, size],
        "tsldst": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(64, size // 4),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(64 * size // 4, size // 4), Stride(16, 4)]),
            ]
        ),
        "tslsrc": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(64 * size // 4, size // 4),
                        Stride(4, 4),
                    ]
                ),
                TiledStride([Stride(64, size // 4), Stride(16, 4)]),
            ]
        ),
        "reshape_var": lambda size: [
            int(size // 4),
            int(size // 4),
            4,
            4,
        ],  # reshape var
        "swapaxis_var": [(0, 1)],  # swap major axis for block transform
    },
    {
        # a transformation like the one needed for gemm
        "name": "transform_gemm",
        "array_sizes": [64, 256, 1024],
        "shape": lambda size: [size, size],
        "tsldst": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(64 * size // 4, size // 4),
                        Stride(16, 4),
                    ]
                ),
                TiledStride([Stride(64, size // 4), Stride(4, 4)]),
            ]
        ),
        "tslsrc": lambda size: TiledStridedLayout(
            [
                TiledStride(
                    [
                        Stride(64 * size // 4, size // 4),
                        Stride(16 * size // 4, 4),
                    ]
                ),
                TiledStride([Stride(16, size // 4), Stride(4, 4)]),
            ]
        ),
        "reshape_var": lambda size: [
            int(size // 4),
            4,
            int(size // 4),
            4,
        ],  # reshape var
        "swapaxis_var": [(1, 2)],  # swap major axis for block transform
    },
]
