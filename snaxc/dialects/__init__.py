from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_snax_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available snax dialects"""

    def get_accfg():
        from snaxc.dialects.accfg import ACCFG

        return ACCFG

    def get_dart():
        from snaxc.dialects.dart import Dart

        return Dart

    def get_debug():
        from snaxc.dialects.test.debug import Debug

        return Debug

    def get_kernel():
        from snaxc.dialects.kernel import Kernel

        return Kernel

    def get_pipeline():
        from snaxc.dialects.pipeline import Pipeline

        return Pipeline

    def get_snax():
        from snaxc.dialects.snax import Snax

        return Snax

    def get_snax_stream():
        from snaxc.dialects.snax_stream import SnaxStream

        return SnaxStream

    def get_tsl():
        from snaxc.dialects.tsl import TSL

        return TSL

    return {
        "accfg": get_accfg,
        "dart": get_dart,
        "debug": get_debug,
        "kernel": get_kernel,
        "pipeline": get_pipeline,
        "snax": get_snax,
        "snax_stream": get_snax_stream,
        "tsl": get_tsl,
    }
