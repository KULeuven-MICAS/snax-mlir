from __future__ import annotations

from typing import cast

from xdsl.dialects.builtin import ArrayAttr, IntegerType, NoneAttr, i32
from xdsl.dialects.llvm import LLVMArrayType, LLVMPointerType, LLVMStructType
from xdsl.dialects.memref import MemRefType, UnrankedMemrefType
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    OpResult,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ClusterSyncOp(IRDLOperation):
    """Cluster sync operation for a snax cluster. This
    translates directly to the C function snrt_cluster_hw_barrier()"""

    name = "snax.cluster_sync_op"


@irdl_op_definition
class LayoutCast(IRDLOperation):
    """LayoutCast operation for memrefs in a snax cluster. This
    operation is used to change the layout of the memref data"""

    name = "snax.layout_cast"

    source = operand_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])
    dest = result_def(MemRefType[Attribute] | UnrankedMemrefType[Attribute])

    def __init__(
        self,
        source: SSAValue | Operation,
        dest: MemRefType[Attribute] | UnrankedMemrefType[Attribute],
    ):
        super().__init__(operands=[source], result_types=[dest])

    @staticmethod
    def from_type_and_target_layout(
        source: SSAValue | Operation,
        layout: Attribute,
    ) -> LayoutCast:
        assert isinstance(source.type, MemRefType)
        dest = MemRefType(
            source.type.get_element_type(),
            shape=source.type.get_shape(),
            layout=layout,
            memory_space=source.type.memory_space,
        )
        return LayoutCast(source, dest)

    def verify_(self) -> None:
        source = cast(MemRefType[Attribute], self.source.type)
        dest = cast(MemRefType[Attribute], self.dest.type)
        if source.get_shape() != dest.get_shape():
            raise VerifyException(
                "Expected source and destination to have the same shape."
            )
        if source.get_element_type() != dest.get_element_type():
            raise VerifyException(
                "Expected source and destination to have the same element type."
            )
        if source.memory_space != dest.memory_space:
            raise VerifyException(
                "Expected source and destination to have the same memory space."
            )


@irdl_op_definition
class Alloc(IRDLOperation):
    """Alloc operation in a snax cluster.

    Contrary to a memref.alloc, this operation does not generate
    a memref. Instead, it returns an llvm struct memref descriptor.
    When other operations get lowerd to llvm, the llvm structs will
    match and the conversion casts can be realized.
    """

    name = "snax.alloc"

    size: Operand = operand_def(IntegerType)
    result: OpResult = result_def(LLVMStructType)
    memory_space: Attribute | None = opt_prop_def(Attribute)

    def __init__(
        self,
        dim: int,
        size: SSAValue | Operation,
        memory_space: Attribute = NoneAttr(),
    ):
        # output type is llvm struct memref descriptor
        result_type = LLVMStructType.from_type_list(
            [
                LLVMPointerType.opaque(),
                LLVMPointerType.opaque(),
                i32,
                LLVMArrayType.from_size_and_type(dim, i32),
                LLVMArrayType.from_size_and_type(dim, i32),
            ]
        )

        super().__init__(
            operands=[size],
            result_types=[result_type],
            properties={"memory_space": memory_space},
        )

    def verify_(self) -> None:
        # check for a correct output type
        if not isinstance(self.result.type, LLVMStructType):
            raise VerifyException("Expected result type to be LLVMStructType")
        if not isinstance(self.result.type.types, ArrayAttr):
            raise VerifyException("Expected result type to have an ArrayAttr")

        type_iter = iter(self.result.type.types.data)

        if not isinstance(next(type_iter), LLVMPointerType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected first element to be LLVMPointerType"
            )
        if not isinstance(next(type_iter), LLVMPointerType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected second element to be LLVMPointerType"
            )
        if not isinstance(next(type_iter), IntegerType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected third element to be IntegerType"
            )
        shape = next(type_iter)
        if not isinstance(shape, LLVMArrayType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected fourth element to be LLVMArrayType"
            )
        if not isinstance(shape.type, IntegerType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected fourth element to be LLVMArrayType of IntegerType"
            )
        strides = next(type_iter)
        if not isinstance(strides, LLVMArrayType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected fifth element to be LLVMArrayType"
            )
        if not isinstance(strides.type, IntegerType):
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected fifth element to be LLVMArrayType of IntegerType"
            )

        if not strides.size.data == shape.size.data:
            raise VerifyException(
                "Invalid Memref Descriptor: "
                + "Expected shape and strides to have the same dimension"
            )


Snax = Dialect("snax", [ClusterSyncOp, LayoutCast, Alloc], [])
