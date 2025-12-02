import shutil
import subprocess
from dataclasses import dataclass
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.utils.exceptions import DiagnosticException


@dataclass(frozen=True)
class PhsExportToVerilogPass(ModulePass):
    """
    Pass that exports MLIR IR to Verilog using the firtool executable from CIRCT.
    Args:
        output (str): Path to the output Verilog file.
    """

    name = "phs-export-to-verilog"

    output: str

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        executable = "firtool"
        if not shutil.which(executable):
            raise ValueError(f"Executable {executable} not found")
        stream = StringIO()
        printer = Printer(print_generic_format=True, stream=stream)
        printer.print_op(op)

        completed_process = subprocess.run(
            [executable, "--format=allooo", "-o", f"{self.output}"],
            input=stream.getvalue(),
            capture_output=False,
            text=True,
        )
        try:
            completed_process.check_returncode()
        except subprocess.CalledProcessError as e:
            raise DiagnosticException(f"Error executing {executable}") from e
