testcases = [
    {
        "name": "equal_layout_dynamic_lcb_4",
        "array_sizes": [64, 256, 1024],
        "shape": lambda _: "?x?",
        "tsldst": lambda _: "[?, 4] -> (16, 4), [?, 4] -> (?, ?)",
        "tslsrc": lambda _: "[?, 4] -> (16, 4), [?, 4] -> (?, ?)",
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "equal_layout_dynamic_lcb_8",
        "array_sizes": [256, 1024],
        "shape": lambda _: "?x?",
        "tsldst": lambda _: "[?, 8] -> (32, 4), [?, 4] -> (?, ?)",
        "tslsrc": lambda _: "[?, 8] -> (32, 4), [?, 4] -> (?, ?)",
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "equal_layout_static",
        "array_sizes": [256, 1024],
        "shape": lambda size: f"{int(size)}x{int(size)}",
        "tsldst": lambda size: f"[{int(size//4)}, 4] -> (16, 4), \
            [{int(size//4)}, 4] -> ({int(16*size)}, {int(16*size//4)})",
        "tslsrc": lambda size: f"[{int(size//4)}, 4] -> (16, 4), \
            [{int(size//4)}, 4] -> ({int(16*size)}, {int(16*size//4)})",
        "reshape_var": None,
        "swapaxis_var": [],
    },
    {
        "name": "transform_block_transform",
        "array_sizes": [64, 256, 1024],
        "shape": lambda size: f"{int(size)}x{int(size)}",
        "tsldst": lambda size: f"[{int(size//4)}, 4] -> (64, 4), \
            [{int(size//4)}, 4] -> ({int(64*size//4)}, 16)",
        "tslsrc": lambda size: f"[{int(size//4)}, 4] -> ({int(64*size//4)}, 4), \
            [{int(size//4)}, 4] -> (64, 16)",
        "reshape_var": lambda size: [
            int(size // 4),
            int(size // 4),
            4,
            4,
        ],  # reshape var
        "swapaxis_var": [(0, 1)],  # swap major axis for block transform
    },
]
