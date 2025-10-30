from xdsl.dialects.arith import AddfOp, DivfOp, MulfOp, SubfOp
from xdsl.dialects.builtin import Float32Type, FunctionType, IndexType
from xdsl.ir import Block, Region

from snaxc.dialects import phs
from snaxc.transforms.phs.combine import append_to_abstract_graph


def test_combine() -> None:
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
            result := phs.ChooseOp.from_operations("0", lhs, rhs, switch, [AddfOp, SubfOp], out_types),
            phs.YieldOp(result),
        ]
    )
    abstract_pe_a = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockA)
    )

    blockB = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockB.args
    blockB.add_ops(
        [
            result := phs.ChooseOp.from_operations("0", lhs, rhs, switch, [MulfOp], out_types),
            phs.YieldOp(result),
        ]
    )
    abstract_pe_b = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockB)
    )

    block_inputs = [*in_types, IndexType(), IndexType()]
    blockC = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2 = blockC.args
    blockC.add_ops(
        [
            result := phs.ChooseOp.from_operations("0", lhs, rhs, switch1, [MulfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("1", lhs, result, switch2, [MulfOp], out_types),
            phs.YieldOp(result_2),
        ]
    )
    abstract_pe_c = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockC)
    )

    block_inputs = [*in_types, IndexType(), IndexType(), IndexType()]
    blockD = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockD.args
    blockD.add_ops(
        [
            result := phs.ChooseOp.from_operations("0", lhs, rhs, switch1, [MulfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("1", lhs, result, switch2, [MulfOp], out_types),
            result_3 := phs.ChooseOp.from_operations("2", result, result_2, switch3, [MulfOp], out_types),
            phs.YieldOp(result_3),
        ]
    )
    abstract_pe_d = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockD)
    )

    blockE = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockE.args
    blockE.add_ops(
        [
            result := phs.ChooseOp.from_operations("0", lhs, rhs, switch1, [AddfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("1", lhs, result, switch2, [AddfOp], out_types),
            result_3 := phs.ChooseOp.from_operations("2", result, rhs, switch3, [DivfOp], out_types),
            phs.YieldOp(result_3),
        ]
    )
    abstract_pe_e = phs.AbstractPEOperation(
        "myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), Region(blockE)
    )
    print("A")
    print(abstract_pe_a)
    print("B")
    print(abstract_pe_b)
    print("A+B")
    append_to_abstract_graph(abstract_pe_a, abstract_pe_b)
    print(abstract_pe_b)
    print("C")
    print(abstract_pe_c)
    print("A+B+C")
    append_to_abstract_graph(abstract_pe_c, abstract_pe_b)
    print(abstract_pe_b)
    print("D")
    print(abstract_pe_d)
    print("A+B+C+D")
    append_to_abstract_graph(abstract_pe_d, abstract_pe_b)
    print(abstract_pe_b)
    print("E")
    print(abstract_pe_e)
    print("A+B+C+D+E")
    append_to_abstract_graph(abstract_pe_e, abstract_pe_b)
    print(abstract_pe_b)
    return


if __name__ == "__main__":
    test_combine()
