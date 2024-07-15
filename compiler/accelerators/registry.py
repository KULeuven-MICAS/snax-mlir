from collections.abc import Iterable

from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.traits import SymbolTable

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.gemmini import GemminiAccelerator
from compiler.accelerators.snax_alu import SNAXAluAccelerator
from compiler.accelerators.snax_gemm import SNAXGEMMAccelerator
from compiler.accelerators.snax_hwpe_mult import SNAXHWPEMultAccelerator
from compiler.dialects.accfg import AcceleratorOp


class AcceleratorRegistry:
    """
    This registry is used to get specific accelerator conversions from
    a query based on an accfg.acceleratorop.
    """

    registered_accelerators = {
        "snax_alu": SNAXAluAccelerator,
        "snax_hwpe_mult": SNAXHWPEMultAccelerator,
        "snax_gemm": SNAXGEMMAccelerator,
        "gemmini": GemminiAccelerator,
    }

    def lookup_acc_info(
        self, acc_query: StringAttr, module: ModuleOp
    ) -> tuple[AcceleratorOp, type[Accelerator] | None]:
        """
        Perform a symbol table lookup for the accelerator op in the IR
        and then get the corresponding the Accelerator interface from
        the accelerator registry.
        Returns both the looked up accelerator op and the Accelerator interface
        """
        trait = module.get_trait(SymbolTable)
        assert trait is not None
        acc_op = trait.lookup_symbol(module, acc_query)
        if not isinstance(acc_op, AcceleratorOp):
            raise RuntimeError(
                f"Symbol Table lookup failed for accelerator '{acc_query.data}'. "
                "Is the symbol declared by an accfg.accelerator op in the module?"
            )
        else:
            return acc_op, self.get_acc_info(acc_op)

    def get_acc_info(
        self, acc_op: AcceleratorOp | StringAttr | str
    ) -> type[Accelerator] | None:
        """
        Get a reference to an Accelerator interface based on an AcceleratorOp,
        string or StringAttr.
        If the requested symbol name is not available, return None
        """
        if isinstance(acc_op, str):
            acc_name = acc_op
        elif isinstance(acc_op, StringAttr):
            acc_name = acc_op.data
        else:
            acc_name = acc_op.name_prop.string_value()
        try:
            acc_info = self.registered_accelerators[acc_name]
        except KeyError:
            return None
        return acc_info

    def get_names(self) -> Iterable[str]:
        """
        Get an Iterable of strings that contains the names of the
        registered accelerators.
        """
        return self.registered_accelerators.keys()
