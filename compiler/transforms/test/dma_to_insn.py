from xdsl.context import MLContext
from xdsl.dialects import arith, builtin, func, llvm, memref
from xdsl.dialects.builtin import i32
from xdsl.dialects.scf import Condition, While, Yield
from xdsl.irdl import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable

from compiler.dialects.test import debug


class DMAToInsn(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Call, rewriter: PatternRewriter):
        if op.callee.root_reference.data != "snax_dma_1d_transfer":
            return

        source = op.operands[0]
        dest = op.operands[1]
        size = op.operands[2]


        source_i32 = arith.IndexCastOp(source, builtin.i32)
        dest_i32 = arith.IndexCastOp(dest, builtin.i32)
        size_i32 = arith.IndexCastOp(size, builtin.i32)

        #f".insn r opcode6, func3, func7, rd, rs1, rs2"
        dmsrc = ".insn r 0b0101011, 0, 0b0000000, x0, $0, x0"
        dmdst = ".insn r 0b0101011, 0, 0b0000001, x0, $0, x0"
        dmcpy = ".insn r 0b0101011, 0, 0b0000010, x0, $0, x0"

        ops_to_insert: list[Operation] = [source_i32, dest_i32, size_i32]
        ops_to_insert.append(llvm.InlineAsmOp(dmsrc, "r", [source_i32], has_side_effects=True))
        ops_to_insert.append(llvm.InlineAsmOp(dmdst, "r", [dest_i32], has_side_effects=True))
        ops_to_insert.append(llvm.InlineAsmOp(dmcpy, "r", [size_i32], has_side_effects=True))

        ops_to_insert.append(zero := arith.Constant.from_int_and_width(0, 32))

        wait_all = While(
            [],
            [],
            [
                status := llvm.InlineAsmOp(
                    ".insn r 0b0101011, 0, 0b0000100, $0, x0, x2",
                    "=r",
                    [],
                    [i32],
                    has_side_effects=True,
                ),
                # check if not equal to zero
                comparison := arith.Cmpi(status, zero, "ne"),
                Condition(comparison.results[0]),
            ],
            [
                Yield(),
            ],
        )

        ops_to_insert.append(wait_all)

        rewriter.replace_matched_op(ops_to_insert)
        # rewriter.insert_op(ops_to_insert, InsertPoint.after(op))


class DMAToInsnPass(ModulePass):
    name = "test-dma-to-insn"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(DMAToInsn(), apply_recursively=False).rewrite_module(op)
