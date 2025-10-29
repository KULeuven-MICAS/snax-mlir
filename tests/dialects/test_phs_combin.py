from os import walk
from typing import Sequence
from numpy import choose
from xdsl.dialects.arith import AddfOp, DivfOp, FloatingPointLikeBinaryOperation, MulfOp, SubfOp
from xdsl.dialects.builtin import Float32Type, FunctionType, IndexType
from xdsl.dialects.scf import YieldOp
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
                "0",
                lhs,
                rhs,
                switch,
                Region(
                    Block(
                        [
                            result := AddfOp(rhs, lhs),
                            phs.YieldOp(result),
                        ]
                    )
                ),
                [Region(Block([result := SubfOp(rhs, lhs), phs.YieldOp(result)]))],
                result_types=out_types,
            ),
            yield_a := phs.YieldOp(result),
        ]
    )
    # printer.print_block(blockA)

    blockB = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockB.args
    blockB.add_ops(
        [
            result := phs.ChooseOpOp(
                "0",
                lhs,
                rhs,
                switch,
                Region(
                    Block(
                        [
                            result := MulfOp(lhs, rhs),
                            phs.YieldOp(result),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            phs.YieldOp(result),
        ]
    )

    block_inputs = [*in_types, IndexType(), IndexType()]
    blockC = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2 = blockC.args
    blockC.add_ops(
        [
            result := phs.ChooseOpOp(
                "0",
                lhs,
                rhs,
                switch1,
                Region(
                    Block(
                        [
                            result_in := MulfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            result_2 := phs.ChooseOpOp(
                "1",
                lhs,
                result,
                switch2,
                Region(
                    Block(
                        [
                            result_in := MulfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            phs.YieldOp(result_2),
        ]
    )
    block_inputs = [*in_types, IndexType(), IndexType(), IndexType()]
    blockD = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockD.args
    blockD.add_ops(
        [
            result := phs.ChooseOpOp(
                "0",
                lhs,
                rhs,
                switch1,
                Region(
                    Block(
                        [
                            result_in := MulfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            result_2 := phs.ChooseOpOp(
                "1",
                lhs,
                result,
                switch2,
                Region(
                    Block(
                        [
                            result_in := MulfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            result_3 := phs.ChooseOpOp(
                "2",
                result,
                result_2,
                switch3,
                Region(
                    Block(
                        [
                            result_in := MulfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            phs.YieldOp(result_3),
        ]
    )

    blockE = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockE.args
    blockE.add_ops(
        [
            result := phs.ChooseOpOp(
                "0",
                lhs,
                rhs,
                switch1,
                Region(
                    Block(
                        [
                            result_in := AddfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            result_2 := phs.ChooseOpOp(
                "1",
                lhs,
                result,
                switch2,
                Region(
                    Block(
                        [
                            result_in := AddfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            result_3 := phs.ChooseOpOp(
                "2",
                result,
                rhs,
                switch3,
                Region(
                    Block(
                        [
                            result_in := DivfOp(lhs, rhs),
                            phs.YieldOp(result_in),
                        ]
                    )
                ),
                result_types=out_types,
            ),
            phs.YieldOp(result_3),
        ]
    )
    # printer.print_block(blockB)
    abstract_pe_a = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockA)
    )
    abstract_pe_b = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockB)
    )
    abstract_pe_c = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockC)
    )
    abstract_pe_d = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockD)
    )
    abstract_pe_e = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockE)
    )
    append_to_abstract_graph(abstract_pe_a, abstract_pe_b)
    append_to_abstract_graph(abstract_pe_c, abstract_pe_b)
    print(abstract_pe_b)
    print(abstract_pe_d)
    append_to_abstract_graph(abstract_pe_d, abstract_pe_b)
    print(abstract_pe_b)
    print(abstract_pe_e)
    print("=" * 20)
    append_to_abstract_graph(abstract_pe_e, abstract_pe_b)
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


def get_equivalent_owner(operand: Operand, abstract_graph: phs.AbstractPEOperation):
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


def get_abstract_possibilities(operand: Operand) -> list[str | int]:
    """
    Get all possible paths on the abstract graph (goes past choose_ops)
    """
    if isinstance(operand.owner, phs.ChooseInputOp):
        possibilities_lhs = get_abstract_possibilities(operand.owner.lhs)
        possibilities_rhs = get_abstract_possibilities(operand.owner.rhs)
        return possibilities_lhs + possibilities_rhs
    elif isinstance(operand, BlockArgument):
        return [operand.index]
    elif isinstance(operand.owner, phs.ChooseOpOp):
        return [operand.owner.name_prop.data]
    else:
        raise NotImplementedError("Only expect owners to be block arguments, ChooseOpOp or ChooseInputOps")


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
    graph: phs.AbstractPEOperation,
    abstract_graph: phs.AbstractPEOperation,
):
    for op in graph.body.ops:
        if isinstance(op, phs.ChooseOpOp):
            choose_op = op
            choose_op_id = choose_op.name_prop.data

            # Get the op with the same id in the other one
            abstract_choose_op = abstract_graph.get_choose_op(choose_op_id)

            # If for this id none exists yet, create a new one and fill it with the operations
            if abstract_choose_op is None:
                # create the abstract_choose_op
                lhs = get_equivalent_owner(choose_op.lhs, abstract_graph)
                assert isinstance(lhs, (BlockArgument, phs.ChooseOpOp))
                rhs = get_equivalent_owner(choose_op.rhs, abstract_graph)
                assert isinstance(rhs, (BlockArgument, phs.ChooseOpOp))
                # Add an extra switch to the PE to control this choice
                switch = abstract_graph.add_extra_switch()
                operations = [type(op) for op in choose_op.operations()]
                abstract_choose_op = phs.ChooseOpOp.from_operations(
                    choose_op_id, lhs, rhs, switch, operations, [Float32Type()]
                )
                abstract_graph.body.block.insert_op_before(abstract_choose_op, abstract_graph.get_terminator())
            # If for this id a choose_op_op exists, make sure the right connections are there
            # then add all the operations that are not yet in the abstract choose_op_op
            else:
                # Make sure all connections are equivalent, otherwise add extra connections
                uncollide_inputs(choose_op, abstract_choose_op, abstract_graph)
                # If all connections are equivalent or muxed, add remaining missing operations
                abstract_choose_op.insert_operations(list(choose_op.operations()))

        # At this level, the only expected YieldOp is the final YieldOp a.k.a. the terminator
        elif isinstance(op, phs.YieldOp):
            # Make sure all connections are equivalent, otherwise add extra connections
            uncollide_inputs(op, abstract_graph.get_terminator(), abstract_graph)

        elif isinstance(op, phs.ChooseInputOp):
            raise NotImplementedError("Don't expect non-abstract input graph to have choose_input ops")
        else:
            raise NotImplementedError("Only expect choose_op_op and yield_op in non-abstract graph")


def uncollide_inputs(
    op: phs.YieldOp | phs.ChooseOpOp, abst_op: phs.YieldOp | phs.ChooseOpOp, abstract_graph: phs.AbstractPEOperation
):
    """
    Check if operations are routed similarly, if they are routed differently,
    add extra inputs with choose_input operations
    """
    # Make sure all connections are equivalent, otherwise add extra connections
    for i, (opnd, abst_opnd) in enumerate(zip(op.data_operands, abst_op.data_operands, strict=True)):
        if are_equivalent(opnd, abst_opnd):
            continue
        else:
            # Add a mux to the switch
            equivalent_owner = get_equivalent_owner(opnd, abstract_graph)
            assert equivalent_owner is not None
            mux = phs.ChooseInputOp(
                lhs=abst_opnd,  # this is the default connection
                rhs=equivalent_owner,  # this is the conflicting connection
                switch=abstract_graph.add_extra_switch(),  # extra switch to control input
                result_types=[Float32Type()],
            )
            abstract_graph.body.block.insert_op_before(mux, abst_op)
            # Reroute the new mux outcome to the abstract terminator
            abst_op.operands[i] = mux.results[0]


if __name__ == "__main__":
    test_combine()
