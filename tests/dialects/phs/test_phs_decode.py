from collections.abc import Sequence

import pytest
from create_input import create_test_input
from xdsl.ir import Block, Operation, Region
from xdsl.printer import Printer

from snaxc.phs.combine import append_to_abstract_graph
from snaxc.phs.decode import decode_to_call_op


def test_decode() -> None:
    pe_a, pe_b, pe_c, pe_d, pe_e, pe_f = create_test_input()
    print("A")
    print(pe_a)
    print("B")
    print(pe_b)
    print("A+B")
    # append_to_abstract_graph(pe_a, pe_b)
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
    print(pe_f)

    printer = Printer()

    def wrap_region(ops: Sequence[Operation]):
        printer.print_region(Region(Block(ops)))

    # PE a is not concrete - it has two operations in its choose_op
    with pytest.raises(AssertionError):
        print("ABSTRACT")
        print(pe_b)
        print("CONCRETE")
        print(pe_a)
        wrap_region(decode_to_call_op(pe_b, pe_a))

    print("ABSTRACT")
    print(pe_b)
    print("CONCRETE")
    print(pe_c)
    wrap_region(decode_to_call_op(pe_b, pe_c))
    print("ABSTRACT")
    print(pe_b)
    print("CONCRETE")
    print(pe_d)
    wrap_region(decode_to_call_op(pe_b, pe_d))
    print("ABSTRACT")
    print(pe_b)
    print("CONCRETE")
    print(pe_e)
    wrap_region(decode_to_call_op(pe_b, pe_e))

    return


if __name__ == "__main__":
    test_decode()
