from os import walk
from typing import Sequence
from numpy import choose
from xdsl.dialects.arith import AddfOp, FloatingPointLikeBinaryOperation, MulfOp, SubfOp
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
                ])),
                [Region(Block([
                    result := SubfOp(rhs, lhs),
                    phs.YieldOp(result)
                ]))],result_types=out_types
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
                    result := MulfOp(lhs, rhs),
                    phs.YieldOp(result),
                ])), result_types=out_types
            ),
            phs.YieldOp(result),
        ]
    )
    blockC = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockC.args
    blockC.add_ops(
        [
            result := phs.ChooseOpOp(
                "0", lhs, rhs, switch, Region(Block([
                    result_in := MulfOp(lhs, rhs),
                    phs.YieldOp(result_in),
                ])), result_types=out_types),
            result_2 := phs.ChooseOpOp(
            "1", lhs, result, switch, Region(Block([
                result_in := MulfOp(lhs, rhs),
                phs.YieldOp(result_in),
            ])), result_types=out_types),
            phs.YieldOp(result_2),
        ]
    )
    blockD = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockD.args
    blockD.add_ops(
        [
            result := phs.ChooseOpOp(
                "0", lhs, rhs, switch, Region(Block([
                    result_in := MulfOp(lhs, rhs),
                    phs.YieldOp(result_in),
                ])), result_types=out_types),
            result_2 := phs.ChooseOpOp(
            "1", lhs, result, switch, Region(Block([
                result_in := MulfOp(lhs, rhs),
                phs.YieldOp(result_in),
            ])), result_types=out_types),
            result_3 := phs.ChooseOpOp(
            "2", result, result_2, switch, Region(Block([
                result_in := MulfOp(lhs, rhs),
                phs.YieldOp(result_in),
            ])), result_types=out_types),
            phs.YieldOp(result_3),
        ]
    )
    #printer.print_block(blockB)
    abstract_pe_a = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockA))
    abstract_pe_b = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockB))
    abstract_pe_c = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockC))
    abstract_pe_d = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockD))
    append_to_abstract_graph(abstract_pe_a, abstract_pe_b)
    append_to_abstract_graph(abstract_pe_c, abstract_pe_b)
    print(abstract_pe_b)
    print(abstract_pe_d)
    append_to_abstract_graph(abstract_pe_d, abstract_pe_b)
    print(abstract_pe_b)

    return

def walk_block_reverse(operation: Operation | Block):
    if not isinstance(operation, Operation):
        return
    else:
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
    # If in the current graph the operand is a BlockArgument
    # return the BlockArgument in the abstract_graph
    if isinstance(operand, BlockArgument):
        return abstract_graph.body.block.args[operand.index]
    # If in the current graph the operand is the result of a previous choice
    # get the same choice block in the abstract graph
    elif isinstance(operand.owner, phs.ChooseOpOp):
        return abstract_graph.get_choose_op(operand.owner.name_prop.data)

def get_abstract_possibilities(operand: Operand, possibilities : list[str | int] = []) -> list[str | int]:
    """
    Get all possible paths on the abstract graph (goes past choose_ops)
    """
    if isinstance(operand.owner, phs.ChooseInputOp):
        possibilities = get_abstract_possibilities(operand.owner.lhs, possibilities)
        possibilities = get_abstract_possibilities(operand.owner.rhs, possibilities)
        return possibilities
    elif isinstance(operand, BlockArgument):
        possibilities.append(operand.index)
        return possibilities
    elif isinstance(operand.owner, phs.ChooseOpOp):
        possibilities.append(operand.owner.name_prop.data)
        return possibilities
    else:
        raise NotImplementedError()


def are_equivalent(operand: Operand, abstract_operand: Operand) -> bool:
    """
    Check if operand of an operation in graph matches path to abstract_graph,
    or any paths exposed by choose_ops
    """
    if isinstance(operand, BlockArgument):
        return any([operand.index == poss for poss in get_abstract_possibilities(abstract_operand)])
    elif isinstance(operand.owner, phs.ChooseOpOp):
        return any([operand.owner.name_prop.data == poss for poss in get_abstract_possibilities(abstract_operand)])
    else:
        return False


def append_to_abstract_graph(
        graph : phs.AbstractPEOperation,
        abstract_graph : phs.AbstractPEOperation,
):
    for op in graph.body.ops:
        if isinstance(op, phs.ChooseOpOp):
            choose_op = op
            choose_op_id = choose_op.name_prop.data
            # Get the similar operation in the other one
            abstract_choose_op = abstract_graph.get_choose_op(choose_op_id)
            if abstract_choose_op is None:
                # create the abstract_choose_op
                lhs = get_equivalent_owner(choose_op.lhs, abstract_graph)
                assert isinstance(lhs, (BlockArgument, phs.ChooseOpOp))
                rhs = get_equivalent_owner(choose_op.rhs, abstract_graph)
                assert isinstance(rhs, (BlockArgument, phs.ChooseOpOp))
                switch = abstract_graph.add_extra_switch()
                operations: Sequence[type[FloatingPointLikeBinaryOperation]] = []
                for op in choose_op.operations():
                    assert isinstance(op, FloatingPointLikeBinaryOperation)
                    operations.append(type(op))

                operations = [type(op) for op in choose_op.operations()]
                abstract_choose_op = phs.ChooseOpOp.from_operations(
                    choose_op_id,
                    lhs,
                    rhs,
                    switch,
                    operations,
                    [Float32Type()]
                )
                abstract_graph.body.block.insert_op_before(abstract_choose_op, abstract_graph.get_terminator())
            else:
                # Make sure all connections are equivalent, otherwise add extra connections
                for opnd, abst_opnd in zip(choose_op.operands, abstract_choose_op.operands,strict=True):
                    if are_equivalent(opnd, abst_opnd):
                        continue
                    else:
                        # Add a mux to the switch
                        raise NotImplementedError()

            abstract_choose_op.insert_operations(list(choose_op.operations()))

        elif isinstance(op, phs.YieldOp):
            # Make sure all connections are equivalent, otherwise add extra connections
            abstract_terminator = abstract_graph.get_terminator()
            for i, (opnd, abst_opnd) in enumerate(zip(op.operands, abstract_terminator.operands, strict=True)):
                if are_equivalent(opnd, abst_opnd):
                    continue
                else:
                    # Add a mux to the switch
                    equivalent_owner = get_equivalent_owner(opnd, abstract_graph)
                    assert equivalent_owner is not None
                    mux = phs.ChooseInputOp(lhs=abst_opnd, # this is the default connection
                                      rhs=equivalent_owner,
                                      switch=abstract_graph.add_extra_switch(),
                                      result_types=[Float32Type()],
                                      )
                    abstract_graph.body.block.insert_op_before(mux,abstract_terminator)
                    abstract_terminator.operands[i] = mux.results[0]


if __name__ == "__main__":
    test_combine()
