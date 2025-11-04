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
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch, [AddfOp, SubfOp], out_types),
            phs.YieldOp(result),
        ]
    )
    pe_a = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 1, Region(blockA))

    blockB = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockB.args
    blockB.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch, [MulfOp], out_types),
            phs.YieldOp(result),
        ]
    )
    pe_b = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 1, Region(blockB))

    block_inputs = [*in_types, IndexType(), IndexType()]
    blockC = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2 = blockC.args
    blockC.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [MulfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [MulfOp], out_types),
            phs.YieldOp(result_2),
        ]
    )
    pe_c = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 2, Region(blockC))

    block_inputs = [*in_types, IndexType(), IndexType(), IndexType()]
    blockD = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockD.args
    blockD.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [MulfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [MulfOp], out_types),
            result_3 := phs.ChooseOp.from_operations("_2", [result, result_2], switch3, [MulfOp], out_types),
            phs.YieldOp(result_3),
        ]
    )
    pe_d = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 3, Region(blockD))

    blockE = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockE.args
    blockE.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [AddfOp], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [AddfOp], out_types),
            result_3 := phs.ChooseOp.from_operations("_2", [result, rhs], switch3, [DivfOp], out_types),
            phs.YieldOp(result_3),
        ]
    )
    pe_e = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 3, Region(blockE))
    print("A")
    print(pe_a)
    print("B")
    print(pe_b)
    print("A+B")
    append_to_abstract_graph(pe_a, pe_b)
    print(pe_b)
    print("C")
    print(pe_c)
    print("A+B+C")
    append_to_abstract_graph(pe_c, pe_b)
    print(pe_b)
    print("D")
    print(pe_d)
    print("A+B+C+D")
    append_to_abstract_graph(pe_d, pe_b)
    print(pe_b)
    print("E")
    print(pe_e)
    print("A+B+C+D+E")
    append_to_abstract_graph(pe_e, pe_b)
    print(pe_b)
    return


if __name__ == "__main__":
    test_combine()
