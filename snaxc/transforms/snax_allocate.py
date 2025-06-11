from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from minimalloc import Buffer, Problem  # pyright: ignore[reportMissingTypeStubs]
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm
from xdsl.dialects.memref import DeallocOp
from xdsl.ir import Operation, OpResult, Sequence, SSAValue
from xdsl.parser import IndexType, IntegerAttr, StringAttr
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.utils.hints import isa

from snaxc.accelerators.acc_context import AccContext
from snaxc.dialects import snax
from snaxc.util.snax_memory import L1, SnaxMemory


def dense_array(pos: Sequence[int] | Sequence[builtin.IntAttr]):
    return builtin.DenseArrayBase.create_dense_int(builtin.i64, pos)


def create_memref_struct(
    alloc_op: snax.Alloc,
    pointer: SSAValue,
    aligned_pointer: SSAValue | None = None,
) -> tuple[Operation, Sequence[Operation]]:
    # default aligned pointer to base pointer
    if aligned_pointer is None:
        aligned_pointer = pointer

    ops_to_insert: list[Operation] = []

    llvm_struct = llvm.UndefOp(alloc_op.result.type)
    ops_to_insert.append(llvm_struct)

    # insert pointer
    llvm_struct = llvm.InsertValueOp(dense_array([0]), llvm_struct.res, pointer)
    ops_to_insert.append(llvm_struct)

    # insert aligned pointer
    llvm_struct = llvm.InsertValueOp(dense_array([1]), llvm_struct.res, aligned_pointer)
    ops_to_insert.append(llvm_struct)

    # insert offset 0
    cst_zero = arith.ConstantOp.from_int_and_width(0, builtin.i32)
    llvm_struct = llvm.InsertValueOp(dense_array([2]), llvm_struct.res, cst_zero.result)
    ops_to_insert.extend([cst_zero, llvm_struct])

    # use shape operands to populate shape of memref descriptor
    for i, shape_op in enumerate(alloc_op.shapes):
        if isinstance(shape_op.type, builtin.IndexType):
            # we must cast to integer for valid llvm op
            shape_op = builtin.UnrealizedConversionCastOp.get([shape_op], [builtin.i32])
            ops_to_insert.append(shape_op)
        else:
            assert isinstance(shape_op, Operation)

        llvm_struct = llvm.InsertValueOp(
            dense_array([3, i]), llvm_struct.res, shape_op.results[0]
        )
        ops_to_insert.append(llvm_struct)

    return llvm_struct, ops_to_insert


class DynamicAllocs(RewritePattern):
    """Swap snax.alloc with function call

    This function implements the snax.allocs
    through C interfacing with function calls. The function call returns a
    pointer to the allocated memory. Aligned allocation is not supported yet,
    and the argument will be ignored. The pass is only implemented for L1 (TCDM)
    memory space allocations.

    In this pass we must also initialize the llvm struct with the correct contents
    for now we only populate pointer, aligned_pointer, offset and shapes.
    The strides are not populated because they are not used, and often not available.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, alloc_op: snax.Alloc, rewriter: PatternRewriter):
        ## only supporting L1 allocation for now
        if alloc_op.memory_space != L1.attribute:
            return

        ops_to_insert: list[Operation] = []

        # create constant alignment op to pass on to the allocation function
        assert isinstance(alloc_op.alignment, builtin.IntegerAttr)
        alignment_op = arith.ConstantOp.from_int_and_width(
            alloc_op.alignment.value.data, builtin.IndexType()
        )
        ops_to_insert.append(alignment_op)

        # call allocation function with size and alignment operations
        func_call = func.CallOp(
            "snax_alloc_l1",
            [alloc_op.size, alignment_op],
            [llvm.LLVMPointerType.opaque()],
        )
        ops_to_insert.append(func_call)

        # the result of the function points to a struct containing 2 pointers:
        # the allocated pointer and the aligned pointer
        func_result_type = llvm.LLVMStructType.from_type_list(
            [
                llvm.LLVMPointerType.opaque(),  # pointer
                llvm.LLVMPointerType.opaque(),  # aligned_pointer
            ]
        )

        # load this struct
        func_result = llvm.LoadOp(func_call.res[0], func_result_type)
        ops_to_insert.append(func_result)

        # extract the allocated pointer and aligned pointer from alloc function call
        pointer_op = llvm.ExtractValueOp(
            dense_array([0]), func_result.results[0], llvm.LLVMPointerType.opaque()
        )
        aligned_pointer_op = llvm.ExtractValueOp(
            dense_array([1]), func_result.results[0], llvm.LLVMPointerType.opaque()
        )
        ops_to_insert.extend([pointer_op, aligned_pointer_op])

        created_struct, ops_to_insert_struct = create_memref_struct(
            alloc_op, pointer_op.res, aligned_pointer_op.res
        )
        ops_to_insert.extend(ops_to_insert_struct)

        module_op = alloc_op.get_toplevel_object()
        assert isinstance(module_op, builtin.ModuleOp)

        rewriter.replace_matched_op(ops_to_insert, created_struct.results)

        func_op = func.FuncOp.external(
            "snax_alloc_l1",
            [builtin.IndexType()] * 2,
            [llvm.LLVMPointerType.opaque()],
        )

        SymbolTable.insert_or_update(module_op, func_op)


@dataclass
class StaticAllocs(RewritePattern):
    """
    Simple static allocation scheme.
    This pass starts allocating at the base pointer of the memory space,
    and keeps incrementing the address by the size of the allocation.
    No allocations are ever freed, so the address space is never reused.
    """

    get_memory: Callable[[str], SnaxMemory]

    current_addresses: dict[SnaxMemory, int] = field(
        default_factory=dict[SnaxMemory, int]
    )

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: snax.Alloc, rewriter: PatternRewriter):
        if not isinstance(op.size, OpResult):
            raise RuntimeError(
                "Static allocations should have a statically known size."
            )
        if not isinstance(op.size.op, arith.ConstantOp):
            raise RuntimeError(
                "Static allocations should have a statically known size."
            )
        if op.memory_space is None:
            raise RuntimeError("Allocations need a defined memory space")

        size_attr = op.size.op.value
        assert isa(size_attr, IntegerAttr[IndexType])
        size = size_attr.value.data

        alignment_attr = op.alignment
        if alignment_attr is None:
            alignment = 0
        else:
            alignment = alignment_attr.value.data

        # get the memory space
        assert isinstance(op.memory_space, builtin.StringAttr)
        memory = self.get_memory(op.memory_space.data)

        if memory not in self.current_addresses:
            self.current_addresses[memory] = memory.start

        current_address = self.current_addresses[memory]

        if current_address % alignment != 0:
            # align the address
            current_address += alignment - (current_address % alignment)

        next_address = current_address + size

        if next_address > memory.start + memory.capacity:
            raise RuntimeError(
                f"Memory space {memory.attribute.data} is full, cannot allocate {size} bytes"
            )

        self.current_addresses[memory] = next_address

        pointer_cst = arith.ConstantOp.from_int_and_width(current_address, builtin.i32)
        pointer = llvm.IntToPtrOp(pointer_cst)

        ops_to_insert: list[Operation] = [pointer_cst, pointer]

        created_struct, ops_to_insert_struct = create_memref_struct(op, pointer.output)
        ops_to_insert.extend(ops_to_insert_struct)

        rewriter.replace_matched_op(ops_to_insert, created_struct.results)


@dataclass
class MiniMallocate(RewritePattern):
    """
    Pattern to assign static allocations based on the minimalloc algorithm.

    For this to work, all allocations must be top level ops in a func op.
    Lifetime is determined by the index of the first and last op it is used in.
    """

    get_memory: Callable[[str], SnaxMemory]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        buffers: list[Buffer] = []
        buffer_ops: dict[str, snax.Alloc] = {}
        uses: dict[Operation, list[Buffer]] = defaultdict(list)

        def get_top_level_op(op: Operation) -> Operation:
            next_op = op
            while next_op.parent_op() is not func_op:
                next_op = next_op.parent_op()
                assert next_op is not None
            return next_op

        if len(func_op.body.blocks) != 1:
            return

        # Determine lifetime of all buffers
        for i, op in enumerate(func_op.body.block.ops):
            if isinstance(op, snax.Alloc):
                # create new buffer
                if not isinstance(op.size, OpResult):
                    raise RuntimeError(
                        "Static allocations should have a statically known size."
                    )
                if not isinstance(op.size.op, arith.ConstantOp):
                    raise RuntimeError(
                        "Static allocations should have a statically known size."
                    )
                if op.memory_space is None:
                    raise RuntimeError("Allocations need a defined memory space")

                size_attr = op.size.op.value
                assert isa(size_attr, IntegerAttr[IndexType])
                size = size_attr.value.data

                alignment_attr = op.alignment
                if alignment_attr is None:
                    alignment = 0
                else:
                    alignment = alignment_attr.value.data

                buffer = Buffer(str(hash(op)), i, i, size, alignment)
                buffers.append(buffer)
                buffer_ops[buffer.id] = op

                # add uses to the use list
                for use in op.results[0].uses:
                    use_op = get_top_level_op(use.operation)
                    uses[use_op].append(buffer)
                    if isinstance(use.operation, builtin.UnrealizedConversionCastOp):
                        for cast_use in use.operation.results[0].uses:
                            cast_use_op = get_top_level_op(cast_use.operation)
                            uses[cast_use_op].append(buffer)

            if op in uses:
                # udpate lifetime of buffer
                for buffer in uses[op]:
                    buffer.end_time = i

        # Insert Dealloc Ops
        last_uses: list[tuple[SSAValue, Operation]] = []
        func_ops = [x for x in func_op.body.block.ops]
        for buffer in buffers:
            memref_op = next(iter(buffer_ops[buffer.id].results[0].uses)).operation
            assert isinstance(memref_op, builtin.UnrealizedConversionCastOp)
            last_uses.append((memref_op.results[0], func_ops[buffer.end_time]))
        for value, op in last_uses:
            rewriter.insert_op(DeallocOp.get(value), InsertPoint.after(op))

        # Lifetime of the buffers is now determined, run the minimalloc algorithm for every memory space
        pointer_result: dict[str, int] = {}
        memory_spaces = set(
            self.get_memory(cast(StringAttr, buffer_ops[buffer.id].memory_space).data)
            for buffer in buffers
        )
        for memory in memory_spaces:
            buffers_subset = [
                buffer
                for buffer in buffers
                if buffer_ops[buffer.id].memory_space == memory.attribute
            ]
            problem = Problem(buffers_subset, memory.capacity)
            solution = problem.solve()
            for buffer, offset in zip(buffers_subset, solution):
                pointer_result[buffer.id] = offset + memory.start

        # Now, generate constant ops for the pointers
        for buffer in buffers:
            pointer_cst = arith.ConstantOp.from_int_and_width(
                pointer_result[buffer.id], builtin.i32
            )
            pointer = llvm.IntToPtrOp(pointer_cst)

            ops_to_insert: list[Operation] = [pointer_cst, pointer]

            created_struct, ops_to_insert_struct = create_memref_struct(
                buffer_ops[buffer.id], pointer.output
            )
            ops_to_insert.extend(ops_to_insert_struct)

            rewriter.replace_op(
                buffer_ops[buffer.id], ops_to_insert, created_struct.results
            )


def allocs_are_static(module_op: builtin.ModuleOp) -> bool:
    """Check if all snax.alloc ops in the module are static."""
    for op in module_op.walk():
        if isinstance(op, snax.Alloc):
            if not isinstance(op.size, OpResult):
                return False
            if not isinstance(op.size.op, arith.ConstantOp):
                return False
    return True


@dataclass(frozen=True)
class SnaxAllocatePass(ModulePass):
    name = "snax-allocate"

    mode: str = "auto"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        if self.mode not in ("dynamic", "static", "minimalloc", "auto"):
            raise RuntimeError(f"unsupported allocation strategy: {self.mode}")
        assert isinstance(ctx, AccContext)
        get_memory = ctx.get_memory
        match self.mode:
            case "dynamic":
                PatternRewriteWalker(DynamicAllocs()).rewrite_module(op)
            case "static":
                PatternRewriteWalker(StaticAllocs(get_memory)).rewrite_module(op)
            case "minimalloc":
                PatternRewriteWalker(MiniMallocate(get_memory)).rewrite_module(op)
            case "auto":
                # auto-detect the mode based on the presence of snax.alloc ops
                if allocs_are_static(op):
                    # use minimalloc for static allocations
                    PatternRewriteWalker(MiniMallocate(get_memory)).rewrite_module(op)
                else:
                    PatternRewriteWalker(DynamicAllocs()).rewrite_module(op)
