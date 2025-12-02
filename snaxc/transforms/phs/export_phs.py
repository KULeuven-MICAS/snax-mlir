from dataclasses import dataclass
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.utils.exceptions import DiagnosticException

from snaxc.dialects import phs


@dataclass(frozen=True)
class PhsExportPhsPass(ModulePass):
    """
    Pass that extracts phs.PE ops, removes them from MLIR and exports them to a new file
    Args:
        output (str): Path to the output MLIR file containing the phs.PE ops.
    """

    name = "phs-export-phs"

    output: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        stream = StringIO()
        printer = Printer(print_generic_format=True, stream=stream)

        ops_to_add: list[phs.PEOp] = []

        for operation in op.ops:
            if isinstance(operation, phs.PEOp):
                operation.detach()
                ops_to_add.append(operation)

        new_module = builtin.ModuleOp(ops_to_add)
        printer.print_op(new_module)
        try:
            with open(self.output, "w") as f:
                f.write(stream.getvalue())
        except OSError as e:
            raise DiagnosticException(f"Failed to write output file {self.output}: {e}") from e
