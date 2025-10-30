from xdsl.dialects.builtin import Float32Type
from xdsl.ir import BlockArgument
from xdsl.irdl import Operand

from snaxc.dialects import phs


def get_equivalent_owner(operand: Operand, abstract_graph: phs.AbstractPEOperation) -> BlockArgument | phs.ChooseOp:
    """
    Get operand of an operation in graph to match to abstract_graph.

    Gives an error if the operand is not the result of a BlockArgument or a ChooseOp,
    Or if a the ChooseOp with the same ID is not found in the abstract graph
    """
    # If in the current graph the operand is a BlockArgument
    # return the BlockArgument in the abstract_graph
    if isinstance(operand, BlockArgument):
        return abstract_graph.body.block.args[operand.index]
    # If in the current graph the operand is the result of a previous choice
    # get the same choice block in the abstract graph
    elif isinstance(operand.owner, phs.ChooseOp):
        abstract_choose_op = abstract_graph.get_choose_op(operand.owner.name_prop.data)
        assert abstract_choose_op is not None, "Equivalent ChooseOp not found in Abstract Graph"
        return abstract_choose_op
    else:
        raise NotImplementedError("Only expect owners to be block arguments or ChooseOps")


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
    elif isinstance(operand.owner, phs.ChooseOp):
        return [operand.owner.name_prop.data]
    else:
        raise NotImplementedError("Only expect owners to be block arguments, ChooseOp or ChooseInputOps")


def are_equivalent(operand: Operand, abstract_operand: Operand) -> bool:
    """
    Check if operand of an operation in graph matches path to abstract_graph,
    or any paths exposed by choose_ops
    """
    if isinstance(operand, BlockArgument):
        return any([operand.index == poss for poss in get_abstract_possibilities(abstract_operand)])
    elif isinstance(operand.owner, phs.ChooseOp):
        return any([operand.owner.name_prop.data == poss for poss in get_abstract_possibilities(abstract_operand)])
    else:
        return False


def uncollide_inputs(op: phs.YieldOp | phs.ChooseOp, abst_op: phs.YieldOp | phs.ChooseOp):
    """
    Check if operations are routed similarly, if they are routed differently,
    add extra inputs with choose_input operations
    """
    # Make sure all connections are equivalent, otherwise add extra connections
    abstract_graph = abst_op.parent_op()
    assert isinstance(abstract_graph, phs.AbstractPEOperation)
    for i, (opnd, abst_opnd) in enumerate(zip(op.data_operands, abst_op.data_operands, strict=True)):
        if are_equivalent(opnd, abst_opnd):
            continue
        else:
            # Add a mux to the switch
            equivalent_owner = get_equivalent_owner(opnd, abstract_graph)
            mux = phs.ChooseInputOp(
                lhs=abst_opnd,  # this is the default connection
                rhs=equivalent_owner,  # this is the conflicting connection
                switch=abstract_graph.add_switch(),  # extra switch to control input
                result_types=[Float32Type()],
            )
            abstract_graph.body.block.insert_op_before(mux, abst_op)
            # Reroute the new mux outcome to the abstract terminator
            abst_op.operands[i] = mux.results[0]


def append_to_abstract_graph(
    graph: phs.AbstractPEOperation,
    abstract_graph: phs.AbstractPEOperation,
):
    """
    Insert graph into abstract_graph such that abstract_graph assumes the capabilities of graph.
    If certain ChooseOp nodes in abstract_graph don't exist they are added.
    If a capability is missing from a ChooseOp, it is added.
    If operations are not routed in abstract_graph the way they are routed in graph,
    ChooseInputOps are inserted automatically to prevent colliding inputs.
    Addition of such a ChooseInputOp adds an extra switch to the abstract_graph's AbstractPEOperation
    """
    for op in graph.body.ops:
        if isinstance(op, phs.ChooseOp):
            choose_op = op
            choose_op_id = choose_op.name_prop.data

            # Get the op with the same id in the other one
            abstract_choose_op = abstract_graph.get_choose_op(choose_op_id)

            # If for this id none exists yet, create a new one and fill it with the operations
            if abstract_choose_op is None:
                # create the abstract_choose_op
                lhs = get_equivalent_owner(choose_op.lhs, abstract_graph)
                rhs = get_equivalent_owner(choose_op.rhs, abstract_graph)
                # Add an extra switch to the PE to control this choice
                switch = abstract_graph.add_switch()
                operations = [type(op) for op in choose_op.operations()]
                abstract_choose_op = phs.ChooseOp.from_operations(
                    choose_op_id, lhs, rhs, switch, operations, [Float32Type()]
                )
                abstract_graph.body.block.insert_op_before(abstract_choose_op, abstract_graph.get_terminator())
            # If for this id a choose_op exists, make sure the right connections are there
            # then add all the operations that are not yet in the abstract choose_op
            else:
                # Make sure all connections are equivalent, otherwise add extra connections
                uncollide_inputs(choose_op, abstract_choose_op)
                # If all connections are equivalent or muxed, add remaining missing operations
                abstract_choose_op.insert_operations(list(choose_op.operations()))

        # At this level, the only expected YieldOp is the final YieldOp a.k.a. the terminator
        elif isinstance(op, phs.YieldOp):
            # Make sure all connections are equivalent, otherwise add extra connections
            uncollide_inputs(op, abstract_graph.get_terminator())

        elif isinstance(op, phs.ChooseInputOp):
            raise NotImplementedError("Don't expect non-abstract input graph to have choose_input ops")
        else:
            raise NotImplementedError("Only expect choose_op and yield_op in non-abstract graph")
