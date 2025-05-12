import pytest
from xdsl.dialects.builtin import IntegerType, MemRefType, i32
from xdsl.dialects.llvm import LLVMArrayType, LLVMPointerType, LLVMStructType
from xdsl.utils.exceptions import VerifyException

from snaxc.util.memref_descriptor import LLVMMemrefDescriptor


def test_llvmmemrefdescriptor_init():
    descriptor = LLVMStructType.from_type_list(
        [
            LLVMPointerType.opaque(),
            LLVMPointerType.opaque(),
            i32,
            LLVMArrayType.from_size_and_type(2, i32),
            LLVMArrayType.from_size_and_type(2, i32),
        ]
    )
    memref_descriptor = LLVMMemrefDescriptor(descriptor)
    assert memref_descriptor.descriptor is descriptor


def test_llvmmemrefdescriptor_from_rank_and_integer_type():
    rank = 2
    integer_type = i32
    memref_descriptor = LLVMMemrefDescriptor.from_rank_and_integer_type(
        rank, integer_type
    )

    assert isinstance(memref_descriptor.descriptor, LLVMStructType)
    assert len(memref_descriptor.descriptor.types.data) == 5

    type_iter = iter(memref_descriptor.descriptor.types.data)

    assert isinstance(next(type_iter), LLVMPointerType)
    assert isinstance(next(type_iter), LLVMPointerType)
    assert isinstance(next(type_iter), IntegerType)
    assert isinstance(next(type_iter), LLVMArrayType)
    assert isinstance(next(type_iter), LLVMArrayType)


def test_llvmmemrefdescriptor_from_memref_type():
    memref_type = MemRefType(i32, [2, 2])
    integer_type = i32
    memref_descriptor = LLVMMemrefDescriptor.from_memref_type(memref_type, integer_type)

    assert isinstance(memref_descriptor.descriptor, LLVMStructType)
    assert len(memref_descriptor.descriptor.types.data) == 5

    type_iter = iter(memref_descriptor.descriptor.types.data)

    assert isinstance(next(type_iter), LLVMPointerType)
    assert isinstance(next(type_iter), LLVMPointerType)
    assert isinstance(next(type_iter), IntegerType)
    assert isinstance(next(type_iter), LLVMArrayType)
    assert isinstance(next(type_iter), LLVMArrayType)


def test_llvmmemrefdescriptor_verify_valid_descriptor():
    descriptor = LLVMStructType.from_type_list(
        [
            LLVMPointerType.opaque(),
            LLVMPointerType.opaque(),
            i32,
            LLVMArrayType.from_size_and_type(2, i32),
            LLVMArrayType.from_size_and_type(2, i32),
        ]
    )

    memref_descriptor = LLVMMemrefDescriptor(descriptor)
    memref_descriptor.verify()


def test_llvmmemrefdescriptor_verify_invalid_descriptor():
    descriptor = LLVMStructType.from_type_list(
        [
            LLVMPointerType.opaque(),
            LLVMPointerType.opaque(),
            i32,
            LLVMArrayType.from_size_and_type(2, i32),
            i32,  # Invalid type
        ]
    )
    memref_descriptor = LLVMMemrefDescriptor(descriptor)
    with pytest.raises(VerifyException):
        memref_descriptor.verify()


def test_llvmmemrefdescriptor_verify_invalid_shape_and_strides():
    descriptor = LLVMStructType.from_type_list(
        [
            LLVMPointerType.opaque(),
            LLVMPointerType.opaque(),
            i32,
            LLVMArrayType.from_size_and_type(2, i32),
            LLVMArrayType.from_size_and_type(3, i32),  # Invalid dimension
        ]
    )
    memref_descriptor = LLVMMemrefDescriptor(descriptor)
    with pytest.raises(VerifyException):
        memref_descriptor.verify()


def test_llvmmemrefdescriptor_verify_invalid_shape_and_strides_type():
    descriptor = LLVMStructType.from_type_list(
        [
            LLVMPointerType.opaque(),
            LLVMPointerType.opaque(),
            i32,
            LLVMArrayType.from_size_and_type(2, i32),
            LLVMArrayType.from_size_and_type(
                2, LLVMPointerType.opaque()
            ),  # Invalid type
        ]
    )
    memref_descriptor = LLVMMemrefDescriptor(descriptor)
    with pytest.raises(VerifyException):
        memref_descriptor.verify()
