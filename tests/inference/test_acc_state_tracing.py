from xdsl.dialects import builtin, scf
from xdsl.ir import BlockArgument

from compiler.dialects import acc
from compiler.inference.trace_acc_state import (infer_state_of,
                                                infer_states_for_if)

ACC = "acc1"


def test_simple_setup_tracing():
    a, b, c = tuple(BlockArgument(builtin.i32, None, i) for i in range(3))

    empty_setup = acc.SetupOp([], [], ACC)

    full_setup = acc.SetupOp([a, b, c], ["A", "B", "C"], ACC, empty_setup)

    assert infer_state_of(empty_setup.out_state) == {}

    assert infer_state_of(full_setup.out_state) == {"A": a, "B": b, "C": c}

    # construct if block that sets state
    if_block = scf.If(
        a,
        [acc.StateType("acc1")],
        [
            s1 := acc.SetupOp([b], ["A"], ACC, full_setup),
            scf.Yield(s1),
        ],
        [
            s2 := acc.SetupOp([c], ["B"], ACC, full_setup),
            scf.Yield(s2),
        ],
    )

    assert infer_states_for_if(if_block, if_block.results[0]) == (
        {"A": b, "B": b, "C": c},  # if block
        {"A": a, "B": c, "C": c},  # else block
    )

    assert infer_state_of(if_block.results[0]) == {"C": c}
