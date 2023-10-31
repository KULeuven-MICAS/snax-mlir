from typing import Any

from xdsl.dialects import linalg
from dialects import linalg_ext
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)


@register_impls
class LinalgExtFunctions(InterpreterFunctions):
    @impl(linalg_ext.Mul)
    def run_mul(
        self, interpreter: Interpreter, op: linalg.Mul, args: tuple[Any, ...]
    ) -> PythonValues:
        if op.library_call is not None:
            raise NotImplementedError(
                "library_call not yet supported in linalg.mul interpreter"
            )
        if op.res:
            raise NotImplementedError(
                "results not yet supported in linalg.mul interpreter"
            )
        return ()
