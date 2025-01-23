from collections.abc import Generator, Sequence


def get_default_paths() -> dict[str, str]:
    return {
        "cc": "clang",
        "ld": "clang",
        "mlir-opt": "mlir-opt",
        "mlir-translate": "mlir-translate",
        "snax-opt": "snax-opt",
    }


def get_traces(
    prefixes: Sequence[str],
    num_chips: int = 1,
    num_harts: int = 2,
    extension: str = "dasm",
) -> Generator[str, None, None]:
    for prefix in prefixes:
        for chip_id in range(num_chips):
            for hart_id in range(num_harts):
                yield f"{prefix}_trace_chip_{chip_id:02d}_hart_{hart_id:05d}.{extension}"


def get_trace_ext(
    prefix: str, num_chips: int = 1, num_harts: int = 2, extension: str = "dasm"
) -> Sequence[str]:
    extensions = [
        trace[: 26 + len(extension)]
        for trace in get_traces([""], num_chips, num_harts, extension)
    ]
    return [prefix, *extensions]


if __name__ == "__main__":
    print(get_trace_ext("file", 1, 2, "dasm"))
