from xdsl.context import Context
from xdsl.dialects import builtin, linalg
from xdsl.dialects.builtin import FunctionType, ModuleOp
from xdsl.ir import Operation
from xdsl.parser import SymbolRefAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable

from snaxc.dialects import phs
from snaxc.transforms.phs.combine import append_to_abstract_graph

MAGIC_ATTR_NAME = "phs_acc"


class EncodeLinalgGeneric(RewritePattern):
    def __init__(self):
        super().__init__()
        # This variable should be new for each instance of this rewritepattern
        self._count: dict[str, int] = {}

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # Bail if this accelerator does not have an acc symbol
        if MAGIC_ATTR_NAME not in linalg_op.attributes:
            return

        # Bail if the acc symbol is not a SymbolRefAttr
        acc_symbol_ref = linalg_op.attributes[MAGIC_ATTR_NAME]
        if not isinstance(acc_symbol_ref, SymbolRefAttr):
            return

        # Convert the linalg body to a phs body in a pe operation
        pe = self.convert_linalg_body_to_phs(linalg_op, acc_symbol_ref.string_value(), rewriter)

        # Get enclosing module_op
        toplevel = linalg_op.get_toplevel_object()
        assert isinstance(toplevel, ModuleOp), "Expect top-level IR object to be a ModuleOp"

        # Check if a PE with the current id exists
        top_table = toplevel.get_trait(SymbolTable)
        assert top_table is not None, "Could not find the top-level symbol table"
        abstract_pe = top_table.lookup_symbol(toplevel, acc_symbol_ref)
        if abstract_pe is None:
            # If a PE with this id does not exist yet, simply insert it
            rewriter.insert_op(pe, InsertPoint.at_start(toplevel.regions[0].block))
        else:
            # If a PE with this id already exists, combine it with the previous
            msg = f"Symbol for {acc_symbol_ref.string_value()} already exists, but is not a PEOp"
            assert isinstance(abstract_pe, phs.PEOp), msg
            append_to_abstract_graph(pe, abstract_pe)

    def convert_linalg_body_to_phs(self, linalg_op: linalg.GenericOp, name: str, rewriter: PatternRewriter) -> phs.PEOp:
        """
        Perform conversion from linalg body -> phs body
        """

        # Get a copy for conversion of the linalg block
        body_copy = linalg_op.body.clone()
        linalg_yield = body_copy.block.ops.last
        assert isinstance(linalg_yield, linalg.YieldOp)

        pe = phs.PEOp(
            name,
            function_type=FunctionType.from_lists(body_copy.block.arg_types, linalg_yield.operand_types),
            switch_no=0,
            region=body_copy,
        )
        for op in pe.body.ops:
            if isinstance(op, linalg.YieldOp):
                yield_op = phs.YieldOp(op.operands[0])
                rewriter.replace_op(op, yield_op)
            else:
                id = self._get_id(op)
                choose_op = phs.ChooseOp.from_operations(
                    id, op.operands, pe.add_switch(), [type(op)], result_types=op.result_types
                )
                rewriter.replace_op(op, choose_op)

        # Reset ids for next encountered linalg
        self._reset_ids()
        return pe

    def _get_id(self, op: Operation):
        """
        Use input and output types to group together operations in similar encoding spaces such that e.g.:
        e.g. the second encountered op with in0 : i32 in1 : i32 and out0 : f32 is be assigned to id
        "i_i32_i32_o_f32"
        """
        key = "i_"
        for opnd in op.operands:
            key += f"{opnd.type}_"

        key += "o_"
        for res in op.results:
            key += f"{res.type}_"

        if key in self._count:
            current_count = self._count[key] + 1
            self._count[key] = current_count
            return key + str(current_count)
        else:
            self._count[key] = 0
            return key + "0"

    def _reset_ids(self):
        """
        Reset all counts for arity assignment
        """
        self._count.clear()


class PhsEncodePass(ModulePass):
    name = "phs-encode"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(EncodeLinalgGeneric(), apply_recursively=False).rewrite_module(op)
