from dataclasses import dataclass
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import DiagnosticException

from snaxc.dialects import phs


def erase_phs(mod: builtin.ModuleOp, rewriter: Rewriter):
    """
    Erase all phs.PEOps
    """
    for operation in mod.ops:
        if isinstance(operation, phs.PEOp):
            rewriter.erase_op(operation)


def keep_phs(mod: builtin.ModuleOp, rewriter: Rewriter):
    """
    Keep only phs.PEOps
    """
    for operation in mod.ops:
        if not isinstance(operation, phs.PEOp):
            rewriter.erase_op(operation)


class PhsKeepPhsPass(ModulePass):
    """
    Pass that removes all operations which are not phs.PE ops
    """

    name = "phs-keep-phs"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        rewriter = Rewriter()
        keep_phs(module, rewriter)


class PhsRemovePhsPass(ModulePass):
    """
    Pass that removes all phs.PE operations
    """

    name = "phs-remove-phs"

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        rewriter = Rewriter()
        erase_phs(module, rewriter)


@dataclass(frozen=True)
class PhsExportPhsPass(ModulePass):
    """
    Pass that extracts phs.PE ops, removes them from MLIR and exports them to a new file
    Args:
        output (str): Path to the output MLIR file containing the phs.PE ops.
    """

    name = "phs-export-phs"

    output: str

    def apply(self, ctx: Context, module: builtin.ModuleOp) -> None:
        stream = StringIO()
        printer = Printer(print_generic_format=True, stream=stream)
        new_module = module.clone()
        rewriter = Rewriter()
        # From the clone, keep only PE ops
        keep_phs(new_module, rewriter)
        # From the original, remove all PE ops
        erase_phs(module, rewriter)
        printer.print_op(new_module)
        try:
            with open(self.output, "w") as f:
                f.write(stream.getvalue())
        except OSError as e:
            raise DiagnosticException(f"Failed to write output file {self.output}: {e}") from e
