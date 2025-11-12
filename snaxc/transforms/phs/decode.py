from collections.abc import Sequence

from xdsl.dialects import arith
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType
from xdsl.ir import Block, BlockArgument, Operation, Region
from xdsl.irdl import Operand

from snaxc.dialects import phs


def decode_abstract_graph(abstract_graph: phs.PEOp, graph: phs.PEOp) -> Region:
    # Make sure the graph to be decoded is concrete
    assert graph.is_concrete(), "Given graph is not concrete, unclear what choices should be made"

    # Both PE's should be part of the same accelerator
    acc_msg = "Expect abstract graph and graph to be tied to same accelerator:\n got {} and {}"
    abstract_graph_name = abstract_graph.name_prop.data
    graph_name = graph.name_prop.data
    assert abstract_graph_name == graph_name, acc_msg.format(abstract_graph_name, graph_name)

    # Get the values for all switches in the PEOp
    call_switches: Sequence[arith.ConstantOp | phs.MuxOp] = []
    mux_switches: Sequence[phs.MuxOp] = []
    for switch in abstract_graph.get_switches():
        switchee = switch.get_user_of_unique_use()
        assert switchee is not None, "Switch drives more than one choice in the PE"
        if isinstance(switchee, phs.ChooseOp):
            equivalent_choice = graph.get_choose_op(switchee.name_prop.data)
            if equivalent_choice is None:
                # The current choose_op is not needed in the graph.
                # Map to default = zero
                call_switches.append(arith.ConstantOp.from_int_and_width(0, IndexType()))
                continue
            assert equivalent_choice is not None
            target_operation = list(equivalent_choice.operations())[0]
            assert isinstance(target_operation, Operation)
            for i, operation in enumerate(switchee.operations()):
                if type(target_operation) is type(operation):
                    call_switches.append(arith.ConstantOp.from_int_and_width(i, IndexType()))
                    break
            else:
                raise RuntimeError(f"Failed to map switch {switch} to a value")

        elif isinstance(switchee, phs.MuxOp):
            # Collect mapping decisions for later mapping
            mux_switches.append(switchee)
            call_switches.append(switchee)
        else:
            call_switches.append(ConstantOp.from_int_and_width(0, IndexType()))

    def follow_operand(operand: Operand, assignment: dict[phs.MuxOp, int]) -> str | int:
        """
        Given a mapping for the muxes, follow the value through the mux operation
        Note: Only use this method on the PE in which the muxes are defined!
        """
        if isinstance(operand, BlockArgument):
            return operand.index
        elif isinstance(operand.owner, phs.ChooseOp):
            return operand.owner.name_prop.data
        elif isinstance(operand.owner, phs.MuxOp):
            if assignment[operand.owner] == 1:
                return follow_operand(operand.owner.rhs, assignment)
            else:
                return follow_operand(operand.owner.lhs, assignment)
        else:
            raise NotImplementedError()

    def valid_mapping(graph: phs.PEOp, abstract_graph: phs.PEOp, assignment: dict[phs.MuxOp, int]) -> bool:
        """
        Check whether all connections made are equivalent
        """
        for operation in graph.body.ops:
            if isinstance(operation, phs.ChooseOp):
                op = operation
                abst_op = abstract_graph.get_choose_op(op.name_prop.data)
                assert abst_op is not None, "Could not find equivalent"
            elif isinstance(operation, phs.YieldOp):
                op = operation
                abst_op = abstract_graph.get_terminator()
            else:
                raise NotImplementedError("Only expect ChooseOps and YieldOps in concrete graph")

            # Check if data_operands are similar, don't care about switches
            for opnd, abst_opnd in zip(op.data_operands, abst_op.data_operands, strict=True):
                # If it is a BlockArgument, check whether the index is the same
                if isinstance(opnd, BlockArgument):
                    if opnd.index == follow_operand(abst_opnd, assignment):
                        continue
                    else:
                        return False
                # If its owner is a choose_op, check whether the name is the same
                if isinstance(opnd.owner, phs.ChooseOp):
                    if opnd.owner.name_prop.data == follow_operand(abst_opnd, assignment):
                        continue
                    else:
                        return False
                else:
                    raise NotImplementedError("Only expect ChooseOps or BlockArgs as operands in concrete graph")
        # If no falses, it must be true?
        return True

    def search_mapping(
        i: int, muxes: Sequence[phs.MuxOp], assignment: dict[phs.MuxOp, int]
    ) -> None | dict[phs.MuxOp, int]:
        if i == len(muxes):
            if valid_mapping(graph, abstract_graph, assignment):
                return assignment.copy()
            return

        mux = muxes[i]
        for choice in (0, 1):
            assignment[mux] = choice
            sol = search_mapping(i + 1, muxes, assignment)
            # If it didn't work
            del assignment[mux]
            if sol is not None:
                return sol

    # Search for the mapping of the switches
    mapping = search_mapping(0, mux_switches, {})
    if mapping is None:
        raise RuntimeError("Could not find valid mapping")

    # Swap MuxOp switches for their actual values
    for i, switch in enumerate(call_switches):
        if isinstance(switch, phs.MuxOp):
            call_switches[i] = arith.ConstantOp.from_int_and_width(mapping[switch], IndexType())

    # Create a new CallOp
    name = abstract_graph.name_prop.data
    data_ops = abstract_graph.data_operands()
    call = phs.CallOp(name, data_ops, call_switches, result_types=abstract_graph.get_terminator().operand_types)
    return Region(Block([*call_switches, call]))
