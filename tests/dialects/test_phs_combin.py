from os import walk
from numpy import choose
from pytest import raises
from xdsl.dialects.arith import AddfOp, FloatingPointLikeBinaryOperation, MulfOp
from xdsl.dialects.builtin import Float32Type, FunctionType, IndexType
from xdsl.ir import Block, BlockArgument, OpOperands, Operation, Region, SSAValue
from xdsl.irdl import Operand
from xdsl.printer import Printer
from xdsl.traits import SymbolTable

from snaxc.dialects import phs


def test_combine() -> None:
    printer = Printer()
    switch_types = [IndexType()]
    # Based on operation
    in_types = [Float32Type(), Float32Type()]
    out_types = [Float32Type()]
    # Construct a new block based on the input of the
    block_inputs = [*in_types, *switch_types]
    blockA = Block(arg_types=block_inputs)
    # Map block args to inputs and outputs to yield
    lhs, rhs, switch = blockA.args
    blockA.add_ops(
        [
            result := phs.ChooseOpOp(
                "0", lhs, rhs, switch, Region(Block([
                    result := AddfOp(rhs, lhs),
                    phs.YieldOp(result),
                ])), result_types=out_types
            ),
            yield_a := phs.YieldOp(result),
        ]
    )
    #printer.print_block(blockA)

    blockB = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockB.args
    blockB.add_ops(
        [
            result := phs.ChooseOpOp(
                "0", lhs, rhs, switch, Region(Block([
                    result := MulfOp(rhs, lhs),
                    phs.YieldOp(result),
                ])), result_types=out_types
            ),
            phs.YieldOp(result),
        ]
    )
    #printer.print_block(blockB)

    #walk_block_reverse(yield_b)
    abstract_pe_a = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockA))
    abstract_pe_b = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockB))
    #print(abstract_pe_a)
    #print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_a, abstract_pe_b)
    print(abstract_pe_b)

    return

def walk_block_reverse(operation: Operation | Block):
    if not isinstance(operation, Operation):
        return
    else:
        print(operation)
        if isinstance(operation, phs.YieldOp):
            walk_block_reverse(operation.arguments[0].owner)
        if isinstance(operation, phs.ChooseOpOp):
            for operand in operation.operands:
                walk_block_reverse(operand.owner)

def get_equivalent_owner(
    operand: Operand,
    abstract_graph: phs.AbstractPEOperation
):
    """
    Get operand of an operation in graph to match to abstract_graph
    """
    if isinstance(operand, BlockArgument):
        return abstract_graph.body.block.args[operand.index]
    elif isinstance(operand.owner, phs.ChooseOpOp):
        return abstract_graph.get_choose_op(str(operand.owner.name_prop))

def are_equivalent(operand: Operand, abstract_operand: Operand) -> bool:
    """
    Check if operand of an operation in graph matches path to abstract_graph
    """
    msg = "abstract operand and regular operand owner type should be equal"
    if isinstance(operand, BlockArgument):
        assert isinstance(abstract_operand, BlockArgument), msg
        return operand.index == abstract_operand.index
    elif isinstance(operand.owner, phs.ChooseOpOp):
        assert isinstance(abstract_operand.owner, phs.ChooseOpOp), msg
        return operand.owner.name_prop == abstract_operand.owner.name_prop
    else:
        return False


def append_to_abstract_graph(
        graph : phs.AbstractPEOperation,
        abstract_graph : phs.AbstractPEOperation,
):
    for choose_op in graph.walk_choose_ops():
        choose_op_id = choose_op.name_prop.data
        # Get the similar operation in the other one
        abstract_choose_op = abstract_graph.get_choose_op(choose_op_id)
        if abstract_choose_op is None:
            # create the abstract_choose_op
            lhs = get_equivalent_owner(choose_op.lhs, abstract_graph)
            assert isinstance(lhs, (BlockArgument, phs.ChooseOpOp))
            rhs = get_equivalent_owner(choose_op.rhs, abstract_graph)
            assert isinstance(rhs, (BlockArgument, phs.ChooseOpOp))
            switch = get_equivalent_owner(choose_op.switch, abstract_graph)
            assert isinstance(switch, (BlockArgument, phs.ChooseOpOp))
            # FIXME, create one from multiple operations?
            operation = list(choose_op.operations())[0]
            assert isinstance(operation, FloatingPointLikeBinaryOperation)
            print(choose_op_id)
            abstract_choose_op = phs.ChooseOpOp.from_operation(
                choose_op_id,
                lhs,
                rhs,
                switch,
                type(operation),
                [Float32Type()]
            )
            abstract_graph.body.block.add_op(abstract_choose_op)
        else:
            # Make sure all connections are equivalent, otherwise add extra connections
            for opnd, abst_opnd in zip(choose_op.operands, abstract_choose_op.operands,strict=True):
                if are_equivalent(opnd, abst_opnd):
                    continue
                else:
                    raise NotImplementedError()
        # Make sure all operations are supported
        for operation in choose_op.operations():
            if operation not in list(abstract_choose_op.operations()):
                assert isinstance(operation, FloatingPointLikeBinaryOperation)
                abstract_choose_op.add_operation(type(operation))




if __name__ == "__main__":
    test_combine()
