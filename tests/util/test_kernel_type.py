import pytest
from xdsl.builder import Builder
from xdsl.dialects import arith, linalg, memref
from xdsl.dialects.builtin import AffineMapAttr, IndexType, IntegerType
from xdsl.ir.affine import AffineMap
from xdsl.utils.test_value import TestSSAValue

from compiler.util.kernel_type import InputException, KernelException, KernelType


@pytest.fixture()
def linalg_operands():
    mem_a = memref.Alloc.get(IndexType(), shape=[8, 8])
    mem_b = memref.Alloc.get(IndexType(), shape=[8, 8])
    mem_c = memref.Alloc.get(IndexType(), shape=[8, 8])
    const_a = arith.Constant(IndexType(), 1)
    const_b = arith.Constant(IndexType(), 1)
    inputs = [mem_a, mem_b, const_a, const_b]
    outputs = [mem_c]

    # indexing maps and iterators do not matter for kernel detection,
    # they are just needed to construct the linalg op
    indexing_map = AffineMap.identity(2)
    indexing_maps = [AffineMapAttr(indexing_map)] * 5
    iterator_types = [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * 2

    return inputs, outputs, indexing_maps, iterator_types


@pytest.fixture()
def linalg_empty(linalg_operands):
    @Builder.implicit_region([IndexType()] * 5)
    def body(args):
        linalg.YieldOp(args[0])

    inputs, outputs, imap, itype = linalg_operands
    linalg_op = linalg.Generic(inputs, outputs, body, imap, itype)
    return linalg_op


@pytest.fixture()
def linalg_mult(linalg_operands):
    @Builder.implicit_region([IndexType()] * 5)
    def body(args):
        a, b = args[0:2]
        d = arith.Muli(a, b)
        linalg.YieldOp(d)

    inputs, outputs, imap, itype = linalg_operands
    linalg_op = linalg.Generic(inputs, outputs, body, imap, itype)
    return linalg_op


@pytest.fixture()
def linalg_mac(linalg_operands):
    @Builder.implicit_region([IndexType()] * 5)
    def body(args):
        a, b = args[0:2]
        acc = args[-1]
        d = arith.Muli(a, b)
        e = arith.Addi(d, acc)
        linalg.YieldOp(e)

    inputs, outputs, imap, itype = linalg_operands
    linalg_op = linalg.Generic(inputs, outputs, body, imap, itype)
    return linalg_op


@pytest.fixture()
def linalg_qmac(linalg_operands):
    @Builder.implicit_region([IndexType()] * 5)
    def body(args):
        a, b, azp, bzp = args[0:4]
        acc = args[-1]
        a_ext = arith.ExtSIOp(a, IndexType())
        b_ext = arith.ExtSIOp(b, IndexType())
        a_zpa = arith.Subi(a_ext, azp)
        b_zpa = arith.Subi(b_ext, bzp)
        b = arith.Muli(a_zpa, b_zpa)
        result = arith.Addi(b, acc)
        linalg.YieldOp(result)

    inputs, outputs, imap, itype = linalg_operands
    linalg_op = linalg.Generic(inputs, outputs, body, imap, itype)
    return linalg_op

@pytest.fixture()
def linalg_clamp(linalg_operands):
    @Builder.implicit_region([IndexType()] * 5)
    def body(args):
        lower = arith.Constant.from_int_and_width(-69, IndexType())
        upper = arith.Constant.from_int_and_width(69, IndexType())
        islower = arith.Cmpi(args[0], lower, 'slt')
        clamped_lower = arith.Select(islower, lower, args[0])
        ishigher = arith.Cmpi(clamped_lower, upper, 'sgt')
        clamped = arith.Select(ishigher, upper, clamped_lower)
        linalg.YieldOp(clamped)
    inputs, outputs, imap, itype = linalg_operands
    linalg_op = linalg.Generic(inputs, outputs, body, imap, itype)
    return linalg_op



def test_parse_mult():
    a = TestSSAValue(IndexType())
    b = TestSSAValue(IndexType())
    mult = arith.Muli(a, b)
    add = arith.Addi(a, b)
    res_a, res_b = KernelType.parse_mult(mult)
    assert res_a is a
    assert res_b is b
    with pytest.raises(KernelException):
        KernelType.parse_mult(add)


def test_parse_add():
    a = TestSSAValue(IndexType())
    b = TestSSAValue(IndexType())
    mult = arith.Muli(a, b)
    add = arith.Addi(a, b)
    res_a, res_b = KernelType.parse_add(add)
    assert res_a is a
    assert res_b is b
    with pytest.raises(KernelException):
        KernelType.parse_add(mult)


def test_parse_zpa():
    a = TestSSAValue(IndexType())
    a_se = arith.ExtSIOp(a, IndexType())
    b = TestSSAValue(IndexType())
    zp1 = arith.Subi(a, b)
    zp2 = arith.Subi(a_se, b)
    res_a, res_b = KernelType.parse_zpa(zp1)
    assert res_a is a
    assert res_b is b
    res_a, res_b = KernelType.parse_zpa(zp2)
    assert res_a is a
    assert res_b is b

    add = arith.Addi(a, b)
    with pytest.raises(KernelException):
        KernelType.parse_zpa(add)


def test_parse_inputs(linalg_mult):
    yield_op, types = KernelType.parse_inputs(linalg_mult)
    assert isinstance(yield_op, linalg.YieldOp)
    assert isinstance(types, dict)
    assert len(types.keys()) == 5
    operands = linalg_mult.body.block.args
    assert isinstance(types[operands[0]], memref.MemRefType)
    assert isinstance(types[operands[1]], memref.MemRefType)
    assert isinstance(types[operands[2]], int)
    assert isinstance(types[operands[3]], int)
    assert isinstance(types[operands[4]], memref.MemRefType)


def test_match_inputs(linalg_mult):
    linalg_op = linalg_mult
    a, b = linalg_op.body.block.args[0:2]
    _, types = KernelType.parse_inputs(linalg_op)
    assert KernelType.match_inputs(a, b, types) is None
    a = TestSSAValue(IndexType())
    b = TestSSAValue(IndexType())
    with pytest.raises(InputException):
        KernelType.match_inputs(a, b, types)


def test_get_type(linalg_empty, linalg_mult, linalg_mac, linalg_qmac, linalg_clamp):
    assert KernelType.get_kernel(linalg_empty) == KernelType.YIELD
    assert KernelType.get_kernel(linalg_mult) == KernelType.MUL
    assert KernelType.get_kernel(linalg_mac) == KernelType.MAC
    assert KernelType.get_kernel(linalg_qmac) == KernelType.QMAC
    assert KernelType.get_kernel(linalg_clamp) == KernelType.CLAMP
