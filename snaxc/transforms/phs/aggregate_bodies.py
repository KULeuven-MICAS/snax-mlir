from dataclasses import dataclass
from xdsl.dialects.builtin import ArrayAttr, Float32Type, IndexType, ModuleOp, SymbolRefAttr, FunctionType
from xdsl.ir import Operation, Region, Block 
from xdsl.passes import ModulePass
from xdsl.context import Context
from xdsl.dialects import linalg, builtin
from xdsl.dialects.arith import Arith, FloatingPointLikeBinaryOperation
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.traits import SymbolTable
from snaxc.dialects import phs





bin_arithops : list[type[Operation]] = [op for op in Arith.operations if issubclass(op, FloatingPointLikeBinaryOperation)]
linalg_yield = [linalg.YieldOp]
allowed_ops : list[type[Operation]] = bin_arithops + linalg_yield

def is_part_of(op : Operation, classes : list[type[Operation]]):
    for operation_class in classes:
        if isinstance(op, operation_class):
            return True
    return False


@dataclass(frozen=True)
class AggregateBodyPattern(RewritePattern):

    module : ModuleOp # FIXME, there are probably better ways to get this module?

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: linalg.GenericOp , rewriter: PatternRewriter):
        assert op.body.first_block is not None


        # Bail if it returns multiple outputs
        if len(op.results) > 1:
            return

        # Bail if not f32
        for arg in op.body.first_block.args:
            if not isinstance(arg.type, Float32Type):
                return

        # Bail if it contains an unsupported operation
        for operation in op.body.first_block.walk():
            if not is_part_of(operation, allowed_ops):
                return

        # Bail if aggregate_to is not given
        if "aggregate_to" not in op.attributes.keys():
            return

        # else, go ahead and add to abstract pe operation
        acc_ref = op.attributes["aggregate_to"]
        assert isinstance(acc_ref, SymbolRefAttr)
        t = self.module.get_trait(SymbolTable)
        assert t is not None
        abstract_pe_op = t.lookup_symbol(self.module, acc_ref)

        if abstract_pe_op is None:
            # Abstract op does not exist yet, so create a new one for the current operation
            operation = op.body.first_block.ops.first
            assert isinstance(operation, FloatingPointLikeBinaryOperation)
            abstract_pe_op = phs.AbstractPEOperation.from_operation(
                acc_ref,
                operation,
            )
            # Add the new one to the symbol table of the module
            t.insert_or_update(self.module, abstract_pe_op)
        else:
            assert isinstance(abstract_pe_op, phs.AbstractPEOperation)
            operation = op.body.first_block.ops.first
            assert isinstance(operation, FloatingPointLikeBinaryOperation)
            choose_op = abstract_pe_op.regions[0].ops.first
            assert isinstance(choose_op, phs.ChooseOpOp)
            if operation not in list(choose_op.operations()):
                choose_op.add_operation(type(operation))


class AggregateBodiesPass(ModulePass):
    name = "aggregate-bodies"
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(AggregateBodyPattern(module=op)).rewrite_module(op)
        return

