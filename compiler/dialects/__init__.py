from collections.abc import Callable

from xdsl.ir import Dialect

from compiler.dialects.test import get_all_snax_test_dialects


def get_all_snax_dialects(
    test_dialects: bool = True,
) -> dict[str, Callable[[], Dialect]]:
    """Returns all available snax dialects.
    If test_dialects is set to True, also return all available test dialects"""

    def get_accfg():
        from compiler.dialects.accfg import ACCFG

        return ACCFG

    def get_dart():
        from compiler.dialects.dart import Dart

        return Dart

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

    snax_dialect_factories = {
        "accfg": get_accfg,
        "dart": get_dart,
        "kernel": get_kernel,
        "snax": get_snax,
        "snax_stream": get_snax_stream,
        "tsl": get_tsl,
    }

    dialect_factories: dict[str, Callable[[], Dialect]] = {}

    if test_dialects:
        dialect_factories.update(get_all_snax_test_dialects())
    dialect_factories.update(snax_dialect_factories)

    return dialect_factories
