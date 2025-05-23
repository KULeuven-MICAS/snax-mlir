import os
import pathlib


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
    trace_to_perfetto = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(),
        "tracing",
        "trace_to_perfetto.py",
    )
    return {
        "python": "python",
        "spike-dasm": "spike-dasm",
        "gen_trace.py": gen_trace_path,
        "trace_to_perfetto": trace_to_perfetto,
    }
