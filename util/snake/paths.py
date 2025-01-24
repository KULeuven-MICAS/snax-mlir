import os


def get_default_paths() -> dict[str, str]:
    return {
        "cc": "clang",
        "ld": "clang",
        "mlir-opt": "mlir-opt",
        "mlir-translate": "mlir-translate",
        "snax-opt": "snax-opt",
    }


def get_default_snax_paths() -> dict[str, str]:
    # use CONDA_PREFIX to access pixi env
    gen_trace_path = f"{os.environ['CONDA_PREFIX']}/snax-utils/gen_trace.py"
    return {
        "python": "python",
        "spike-dasm": "spike-dasm",
        "gen_trace.py": gen_trace_path,
    }
