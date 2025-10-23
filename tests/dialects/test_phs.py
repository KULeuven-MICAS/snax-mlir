from xdsl.dialects.builtin import Float32Type, IndexType
from xdsl.dialects.test import TestOp
from xdsl.dialects.arith import AddfOp, MulfOp, DivfOp
from xdsl.ir import Region, Block
from snaxc.dialects.phs import ChooseOpOp, ChooseInputOp, YieldOp
from xdsl.printer import Printer


def test_phs():
    block = Block([
        a := TestOp([],[Float32Type()]),
        b := TestOp([],[Float32Type()]),
        c := TestOp([],[Float32Type()]),
        switch2 := TestOp([],[IndexType()]),
        switch := TestOp([],[IndexType()]),
        chooseinput := ChooseInputOp(b,c, switch2, [Float32Type()]),
        chooseop := ChooseOpOp(a, chooseinput, switch,
                            Region(Block([
            addf := AddfOp(a, chooseinput),
            YieldOp(addf)
        ])),[],[Float32Type()])
    ])
    chooseop.add_operation(MulfOp)
    chooseop.add_operation(DivfOp)
    p = Printer()
    p.print_block(block)


if __name__ == "__main__":
    test_phs()
