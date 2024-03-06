from abc import ABC
from dataclasses import dataclass
from functools import cache

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import MLContext, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import SymbolTable

from compiler.dialects import acc


@dataclass
class LowerAccPattern(RewritePattern, ABC):
    """
    Wraps some common logic to get handles to accelerator ops inside the module.
    """

    module: builtin.ModuleOp

    @cache
    def get_acc(self, accelerator: StringAttr) -> acc.AcceleratorOp:
        """
        Get a reference to the accelerator
        """
        trait = self.module.get_trait(SymbolTable)
        assert trait is not None
        acc_op = trait.lookup_symbol(self.module, accelerator)
        if not isinstance(acc_op, acc.AcceleratorOp):
            raise RuntimeError(
                f"Invalid IR: no accelerator op for @{accelerator.data} found in module"
            )
        return acc_op

    def __hash__(self):
        return id(self)


class LowerAccSetupToCsr(LowerAccPattern):
    """
    Convert setup ops to a series of CSR sets:
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.SetupOp, rewriter: PatternRewriter, /):
        # grab a dict that translates field names to CSR addresses:
        field_to_csr = dict(self.get_acc(op.accelerator).field_items())

        # emit the llvm assembly code to set csr values:
        for field, val in op.iter_params():
            addr = field_to_csr[field]
            rewriter.insert_op_before_matched_op(
                [
                    addr_val := arith.Constant(addr),
                    llvm.InlineAsmOp(
                        [[addr_val, val]],
                        [[]],
                        "csrw $0, $1",
                        "I, r",
                        has_side_effects=True,
                    ),
                ]
            )
        # delete the old setup op
        rewriter.erase_matched_op(safe_erase=False)


class LowerAccLaunchToCsr(LowerAccPattern):
    """
    Convert launch ops to a single csr set (1)
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.LaunchOp, rewriter: PatternRewriter, /):
        assert isinstance(op.state.type, acc.StateType)
        acc_op = self.get_acc(op.state.type.accelerator)

        # insert an op that sets the launch CSR to 1
        rewriter.replace_matched_op(
            [
                addr_val := arith.Constant(acc_op.launch_addr),
                val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
                llvm.InlineAsmOp(
                    [[addr_val, val]],
                    [[]],
                    "csrw $0, $1",
                    # I = any 12 bit immediate, K = any 5 bit immediate
                    # The K allows LLVM to emit an `csrrwi` instruction,
                    # which has room for one 5 bit immediate only.
                    "I, K",
                    has_side_effects=True,
                ),
            ],
            [op.state],
            safe_erase=False,
        )


class LowerAccAwaitToCsr(LowerAccPattern):
    """
    Lower await ops to a series of CSR sets:

    1: enable barriers
    2: trigger barrier
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.AwaitOp, rewriter: PatternRewriter, /):
        assert isinstance(op.token.type, acc.StateType | acc.TokenType)
        acc_op = self.get_acc(op.token.type.accelerator)

        # TODO: this is a temporary solution that will be reworked eventually

        # emit a barrier_enable, and then a barrier_trigger:
        rewriter.replace_matched_op(
            [
                barrier_enable := arith.Constant(acc_op.barrier_enable),
                barrier_trigger := arith.Constant(acc_op.barrier_trigger),
                one := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
                zero := arith.Constant(builtin.IntegerAttr.from_int_and_width(0, 5)),
                llvm.InlineAsmOp(
                    [[barrier_enable, one]],
                    [[]],
                    "csrw $0, $1",
                    # I = any 12 bit immediate, K = any 5 bit immediate
                    # The K allows LLVM to emit an `csrrwi` instruction,
                    # which has room for one 5 bit immediate only.
                    "I, K",
                    has_side_effects=True,
                ),
                llvm.InlineAsmOp(
                    [[barrier_trigger, zero]],
                    [[]],
                    "csrw $0, $1",
                    # I = any 12 bit immediate, K = any 5 bit immediate
                    # The K allows LLVM to emit an `csrrwi` instruction,
                    # which has room for one 5 bit immediate only.
                    "I, K",
                    has_side_effects=True,
                ),
            ],
            safe_erase=False,
        )


class DeleteAllStates(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if any(isinstance(operand.type, acc.StateType) for operand in op.operands):
            new_op = op.__class__.create(
                operands=[
                    operand
                    for operand in op.operands
                    if not isinstance(operand.type, acc.StateType)
                ],
                result_types=[res.type for res in op.results],
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=[reg.clone() for reg in op.regions],
            )
            rewriter.replace_op(op, new_op)
            op = new_op

        if any(isinstance(result.type, acc.StateType) for result in op.results):
            new_op = op.__class__.create(
                operands=op.operands,
                result_types=[
                    res.type
                    for res in op.results
                    if not isinstance(res.type, acc.StateType)
                ],
                properties=op.properties,
                attributes=op.attributes,
                successors=op.successors,
                regions=[op.detach_region(reg) for reg in tuple(op.regions)],
            )
            new_ops_results = list(new_op.results)[::-1]
            replace_results_by = [
                None if isinstance(res.type, acc.StateType) else new_ops_results.pop()
                for res in op.results
            ]
            rewriter.replace_op(op, new_op, new_results=replace_results_by)


class RemoveAcceleratorOps(RewritePattern):
    """
    Delete all accelerator ops after we lowered the setup ops
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: acc.AcceleratorOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


class ConvertAccToCsrPass(ModulePass):
    """
    Converts acc2 dialect ops to series of SNAX-like csr sets.
    """

    name = "convert-acc-to-csr"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccSetupToCsr(op),
                    LowerAccLaunchToCsr(op),
                    LowerAccAwaitToCsr(op),
                    DeleteAllStates(),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)

        PatternRewriteWalker(RemoveAcceleratorOps()).rewrite_module(op)
