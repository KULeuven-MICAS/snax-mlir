from xdsl.dialects.builtin import IntegerType, MemRefType, Signedness
from xdsl.dialects.llvm import LLVMArrayType, LLVMPointerType, LLVMStructType
from xdsl.ir import Attribute
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

# this file contains useful helper functions to work with
# llvm struct memref descriptors, which are used for lowering
# memrefs to llvm ir. The memref descriptor is a struct that
# contains all the necessary information to access a memref
# in memory.

# See https://mlir.llvm.org/docs/TargetLLVMIR/#default-calling-convention-for-ranked-memref
# for more information on the layout of the memref descriptor.


class LLVMMemrefDescriptor:
    """A class to work with LLVM memref descriptors.
    https://mlir.llvm.org/docs/TargetLLVMIR/#default-calling-convention-for-ranked-memref
    """

    descriptor: LLVMStructType

    def __init__(self, descriptor: LLVMStructType):
        """
        Initializes a new instance of the MemrefDescriptor class.

        Args:
            descriptor (LLVMStructType): The descriptor for the memref.
        """
        self.descriptor = descriptor

    @classmethod
    def from_rank_and_integer_type(cls, rank: int, integer_type: IntegerType) -> "LLVMMemrefDescriptor":
        """
        Create an LLVMMemrefDescriptor from a dimension and an integer type.

        Args:
            rank (int): The rank of the memref.
            integer_type (IntegerType): The integer type of the memref.

        Returns:
            LLVMMemrefDescriptor: The created descriptor.
        """

        return cls(
            LLVMStructType.from_type_list(
                [
                    LLVMPointerType(),
                    LLVMPointerType(),
                    integer_type,
                    LLVMArrayType.from_size_and_type(rank, integer_type),
                    LLVMArrayType.from_size_and_type(rank, integer_type),
                ]
            )
        )

    @classmethod
    def from_memref_type(cls, memref_type: MemRefType[Attribute], integer_type: IntegerType) -> "LLVMMemrefDescriptor":
        """
        Create an LLVMMemrefDescriptor from a MemRefType.

        Args:
            memref_type (MemRefType): The MemRefType to create the descriptor from.
            integer_type (IntegerType): The integer type for the memref descriptor.

        Returns:
            LLVMMemrefDescriptor: The created descriptor.
        """

        el_type = memref_type.get_element_type()
        assert isa(el_type, IntegerType[int, Signedness])

        return cls.from_rank_and_integer_type(memref_type.get_num_dims(), el_type)

    def verify(self) -> None:
        """
        Verify the validity of a memref descriptor.

        Raises:
            VerifyException: If the memref descriptor is invalid.
        """

        def exception(message: str) -> VerifyException:
            return VerifyException("Invalid Memref Descriptor: " + message)

        type_iter = iter(self.descriptor.types.data)

        if not isinstance(next(type_iter), LLVMPointerType):
            raise exception("Expected first element to be LLVMPointerType")

        if not isinstance(next(type_iter), LLVMPointerType):
            raise exception("Expected second element to be LLVMPointerType")

        if not isinstance(next(type_iter), IntegerType):
            raise exception("Expected third element to be IntegerType")

        shape = next(type_iter)
        if not isinstance(shape, LLVMArrayType):
            raise exception("Expected fourth element to be LLVMArrayType")

        if not isinstance(shape.type, IntegerType):
            raise exception("Expected fourth element to be LLVMArrayType of IntegerType")

        strides = next(type_iter)
        if not isinstance(strides, LLVMArrayType):
            raise exception("Expected fifth element to be LLVMArrayType")

        if not isinstance(strides.type, IntegerType):
            raise exception("Expected fifth element to be LLVMArrayType of IntegerType")

        if not strides.size.data == shape.size.data:
            raise exception("Expected shape and strides to have the same dimension")
