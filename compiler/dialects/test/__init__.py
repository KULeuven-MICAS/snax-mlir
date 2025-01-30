from collections.abc import Callable

from xdsl.ir import Dialect


def get_all_snax_test_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available snax test dialects."""

    def get_debug():
        from compiler.dialects.test.debug import Debug

        return Debug

    return {
        "debug": get_debug,
    }
