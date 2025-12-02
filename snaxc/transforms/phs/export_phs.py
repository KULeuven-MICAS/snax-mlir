from dataclasses import dataclass
from io import StringIO

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass
from xdsl.printer import Printer

from snaxc.dialects import phs


@dataclass(frozen=True)
class PhsExportPhsPass(ModulePass):
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
        with open(self.output, "w") as f:
            f.write(stream.getvalue())
