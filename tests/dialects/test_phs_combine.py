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
            phs.YieldOp(result),
        ]
    )
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
    print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_a, abstract_pe_b)
    print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_d, abstract_pe_b)
    print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_c, abstract_pe_b)
    print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_d, abstract_pe_b)
    print(abstract_pe_b)
    append_to_abstract_graph(abstract_pe_e, abstract_pe_b)
    print(abstract_pe_b)
    return


if __name__ == "__main__":
    test_combine()
