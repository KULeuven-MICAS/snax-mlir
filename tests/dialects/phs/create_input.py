from xdsl.dialects.arith import AddfOp, AddiOp, DivfOp, MulfOp, MuliOp, SubfOp
from xdsl.dialects.builtin import Float32Type, FunctionType, IndexType, i32
from xdsl.ir import Block, Region

from snaxc.dialects import phs


def create_test_input():
    switch_types = [IndexType()]
    # Based on operation
    in_types = [Float32Type(), Float32Type()]
    out_types = [Float32Type()]
    # Construct a new block based on the input of the
    block_inputs = [*in_types, *switch_types]
    blockA = Block(arg_types=block_inputs)
    # Map block args to inputs and outputs to yield
    lhs, rhs, switch = blockA.args
    addf_op = AddfOp(lhs, rhs)
    subf_op = SubfOp(lhs, rhs)
    blockA.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch, [addf_op, subf_op], out_types),
            phs.YieldOp(result),
        ]
    )
    pe_a = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 1, Region(blockA))

    blockB = Block(arg_types=block_inputs)
    lhs, rhs, switch = blockB.args
    mulf_op = MulfOp(lhs, rhs)
    blockB.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch, [mulf_op], out_types),
            phs.YieldOp(result),
        ]
    )
    pe_b = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 1, Region(blockB))

    block_inputs = [*in_types, IndexType(), IndexType()]
    blockC = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2 = blockC.args
    mulf_op = MulfOp(lhs, rhs)
    mulf2_op = MulfOp(lhs, mulf_op)
    blockC.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [mulf_op], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [mulf2_op], out_types),
            phs.YieldOp(result_2),
        ]
    )
    pe_c = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 2, Region(blockC))

    block_inputs = [*in_types, IndexType(), IndexType(), IndexType()]
    blockD = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockD.args
    mulf_op = MulfOp(lhs, rhs)
    mulf2_op = MulfOp(lhs, mulf_op)
    mulf3_op = MulfOp(mulf_op, mulf2_op)
    blockD.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [mulf_op], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [mulf2_op], out_types),
            result_3 := phs.ChooseOp.from_operations("_2", [result, result_2], switch3, [mulf3_op], out_types),
            phs.YieldOp(result_3),
        ]
    )
    pe_d = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 3, Region(blockD))

    blockE = Block(arg_types=block_inputs)
    lhs, rhs, switch1, switch2, switch3 = blockE.args
    addf_op = AddfOp(lhs, rhs)
    addf2_op = AddfOp(lhs, addf_op)
    divf_op = DivfOp(addf2_op, rhs)
    blockE.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [addf_op], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [addf2_op], out_types),
            result_3 := phs.ChooseOp.from_operations("_2", [result, rhs], switch3, [divf_op], out_types),
            phs.YieldOp(result_3),
        ]
    )
    pe_e = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 3, Region(blockE))

    blockF = Block(arg_types=[i32, i32, IndexType(), IndexType(), IndexType()])
    lhs, rhs, switch1, switch2, switch3 = blockF.args
    addi_op = AddiOp(lhs, rhs)
    addi2_op = AddiOp(lhs, addi_op)
    muli_op = MuliOp(addi_op, rhs)
    out_types = [i32]
    blockF.add_ops(
        [
            result := phs.ChooseOp.from_operations("_0", [lhs, rhs], switch1, [addi_op], out_types),
            result_2 := phs.ChooseOp.from_operations("_1", [lhs, result], switch2, [addi2_op], out_types),
            result_3 := phs.ChooseOp.from_operations("_2", [result, rhs], switch3, [muli_op], out_types),
            phs.YieldOp(result_3),
        ]
    )
    pe_f = phs.PEOp("myfirstaccelerator", FunctionType.from_lists(block_inputs, out_types), 3, Region(blockF))
    return pe_a, pe_b, pe_c, pe_d, pe_e, pe_f
