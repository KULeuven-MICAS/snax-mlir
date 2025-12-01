import pytest
from create_input import create_test_input

from snaxc.phs.combine import append_to_abstract_graph


def test_combine() -> None:
    pe_a, pe_b, pe_c, pe_d, pe_e, pe_f = create_test_input()
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
    # This will try to add i32 operands to a choose_op with f32 operands
    with pytest.raises(AssertionError):
        append_to_abstract_graph(pe_f, pe_b)
    print(pe_b)
    print(pe_f)
    return


if __name__ == "__main__":
    test_combine()
