from os import walk
from xdsl.dialects.arith import AddfOp
from xdsl.dialects.builtin import Float32Type, FunctionType, IndexType
from xdsl.ir import Block, Operation, Region
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
            phs.YieldOp(result),
        ]
    )
    printer.print_block(blockA)

    blockB = Block(arg_types=block_inputs)
    blockB.add_ops(
        [
            result := phs.ChooseOpOp(
                "0", lhs, rhs, switch, Region(Block([
                    result := AddfOp(rhs, lhs),
                    phs.YieldOp(result),
                ])), result_types=out_types
            ),
            yield_b := phs.YieldOp(result),
        ]
    )
    printer.print_block(blockB)

    walk_block_reverse(yield_b)
    abstract_pe = phs.AbstractPEOperation("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockA))
    print(abstract_pe.get_choose_op("0"))
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


def append_to_abstract_graph(graph : Block, abstract_graph : Block, start_node : Operation, visited : None | set[Operation] = None):
    if visited is None:
        visited = set()

    if start_node in visited:
        return

    #visited.add(start_node):

    ##if isinstance 
        


if __name__ == "__main__":
    test_combine()
