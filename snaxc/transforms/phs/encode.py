from xdsl.context import Context
from xdsl.dialects import arith, builtin, linalg
from xdsl.dialects.builtin import AffineMapAttr, FunctionType, ModuleOp, TensorType
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.parser import DenseIntOrFPElementsAttr, Float32Type, SymbolRefAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable

from snaxc.dialects import phs
from snaxc.transforms.phs.combine import append_to_abstract_graph

MAGIC_ATTR_NAME = "phs_acc"


class EncodeLinalgGeneric(RewritePattern):
    _count: dict[str, int] = {}

    @op_type_rewrite_pattern
    def match_and_rewrite(self, linalg_op: linalg.GenericOp, rewriter: PatternRewriter):
        # Bail if this accelerator does not have an acc symbol
        if MAGIC_ATTR_NAME not in linalg_op.attributes:
            return

        # Bail if the acc symbol is not a SymbolRefAttr
        acc_symbol_ref = linalg_op.attributes[MAGIC_ATTR_NAME]
        if not isinstance(acc_symbol_ref, SymbolRefAttr):
            return

        # Get a copy for conversion of the linalg block
        body_copy = linalg_op.body.clone()
        linalg_yield = body_copy.block.ops.last
        assert isinstance(linalg_yield, linalg.YieldOp)

        # Perform conversion from linalg body -> phs body
        pe = phs.PEOp(
            acc_symbol_ref.string_value(),
            function_type=FunctionType.from_lists(body_copy.block.arg_types, linalg_yield.operand_types),
            switch_no=0,
            region=body_copy,
        )
        for op in pe.body.ops:
            if isinstance(op, arith.FloatingPointLikeBinaryOperation):
                choose_op = phs.ChooseOp.from_operations(
                    self._get_id(op), op.lhs, op.rhs, pe.add_switch(), [type(op)], result_types=[Float32Type()]
                )
                rewriter.replace_op(op, choose_op)
            elif isinstance(op, linalg.YieldOp):
                yield_op = phs.YieldOp(op.operands[0])
                rewriter.replace_op(op, yield_op)
            else:
                raise NotImplementedError()

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
            msg = f"Symbol for {acc_symbol_ref.string_value} already exists, but is not a PEOp"
            assert isinstance(abstract_pe, phs.PEOp), msg
            append_to_abstract_graph(pe, abstract_pe)

        # Reset ids for next encountered linalg
        self._reset_ids()

    def _get_id(self, op: Operation):
        """
        Use Arity to group together operations in similar encoding spaces such that e.g.:
        * the second encountered binary op will be assigned to id "_2_1"
        * the first encountered ternary op is assigned to id "_3_0"
        * ...
        """
        # TODO, add types to this assignment?
        arity = len(op.operands)
        key = f"_{arity}_opnd"
        if key in self._count:
            current_count = self._count[key] + 1
            self._count[key] = current_count
            return key + "_" + str(current_count)
        else:
            self._count[key] = 0
            return key + "_0"

    def _reset_ids(self):
        """
        Reset all counts for arity assignment
        """
        self._count = {}


class PhsEncodePass(ModulePass):
    name = "phs-encode"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(EncodeLinalgGeneric(), apply_recursively=False).rewrite_module(op)


if __name__ == "__main__":

    def create_generic() -> linalg.GenericOp:
        # Inputs
        tensor_type = TensorType(element_type=Float32Type(), shape=[1, 3])
        tensor_val_type = DenseIntOrFPElementsAttr.from_list(type=tensor_type, data=[1.0, 2.0, 3.1])
        tensor_cst = arith.ConstantOp(tensor_val_type)

        # Body creation
        body_block = Block(arg_types=[Float32Type(), Float32Type()])
        rhs, lhs = body_block.args

        ops = [
            added := arith.AddfOp(rhs, lhs),
            final_add := arith.AddfOp(
                lhs,
                added,
            ),
            linalg.YieldOp(final_add),
        ]
        body_block.add_ops(ops)
        body = Region(body_block)

        # Affine Map Creation
        d0 = AffineExpr.dimension(0)
        d1 = AffineExpr.dimension(1)

        indexing_map = AffineMap(num_dims=2, num_symbols=0, results=(d0, d1))
        im_attr = AffineMapAttr(indexing_map)

        generic = linalg.GenericOp(
            inputs=[tensor_cst.result, tensor_cst.result],
            outputs=[],
            body=body,
            indexing_maps=[im_attr, im_attr, im_attr],
            iterator_types=[
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
            ],
            result_types=[tensor_type],
        )
        generic.attributes.update({"acc": SymbolRefAttr("accelerator0")})
        return generic

    def create_generic2() -> linalg.GenericOp:
        # Inputs
        tensor_type = TensorType(element_type=Float32Type(), shape=[1, 3])
        tensor_val_type = DenseIntOrFPElementsAttr.from_list(type=tensor_type, data=[1.0, 2.0, 3.1])
        tensor_cst = arith.ConstantOp(tensor_val_type)

        # Body creation
        body_block = Block(arg_types=[Float32Type(), Float32Type()])
        rhs, lhs = body_block.args

        ops = [
            added := arith.MulfOp(rhs, lhs),
            final_add := arith.AddfOp(
                lhs,
                added,
            ),
            linalg.YieldOp(final_add),
        ]
        body_block.add_ops(ops)
        body = Region(body_block)

        # Affine Map Creation
        d0 = AffineExpr.dimension(0)
        d1 = AffineExpr.dimension(1)

        indexing_map = AffineMap(num_dims=2, num_symbols=0, results=(d0, d1))
        im_attr = AffineMapAttr(indexing_map)

        generic = linalg.GenericOp(
            inputs=[tensor_cst.result, tensor_cst.result],
            outputs=[],
            body=body,
            indexing_maps=[im_attr, im_attr, im_attr],
            iterator_types=[
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
                linalg.IteratorTypeAttr.parallel(),
            ],
            result_types=[tensor_type],
        )
        generic.attributes.update({MAGIC_ATTR_NAME: SymbolRefAttr("accelerator0")})
        return generic

    def encode_generic(module_op: ModuleOp) -> ModuleOp:
        PatternRewriteWalker(EncodeLinalgGeneric(), apply_recursively=False).rewrite_module(module_op)
        return module_op

    generic = create_generic()
    module_op = ModuleOp([create_generic(), create_generic2()])
    print(module_op)
    module_op = encode_generic(module_op)
    print(module_op)
