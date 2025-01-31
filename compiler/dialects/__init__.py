from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_snax_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available snax dialects"""

    def get_accfg():
        from compiler.dialects.accfg import ACCFG

        return ACCFG

    def get_dart():
        from compiler.dialects.dart import Dart

        return Dart

    def get_debug():
        from compiler.dialects.test.debug import Debug

        return Debug

    def get_kernel():
        from compiler.dialects.kernel import Kernel

        return Kernel

    def get_snax():
        from compiler.dialects.snax import Snax

        return Snax

    def get_snax_stream():
        from compiler.dialects.snax_stream import SnaxStream

        return SnaxStream

    def get_tsl():
        from compiler.dialects.tsl import TSL

        return TSL

    return {
        "accfg": get_accfg,
        "dart": get_dart,
        "debug": get_debug,
        "kernel": get_kernel,
        "snax": get_snax,
        "snax_stream": get_snax_stream,
        "tsl": get_tsl,
    }
