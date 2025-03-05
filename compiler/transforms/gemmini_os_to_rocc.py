from abc import ABC
from dataclasses import dataclass
from functools import cache

from compiler.accelerators.gemmini_os import GemminiMvoutAccelerator, GemminiExAccelerator, GemminiMvinAccelerator
from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.accelerators.accelerator import Accelerator
from compiler.accelerators.registry import AcceleratorRegistry
from compiler.dialects import accfg
from compiler.transforms.convert_accfg_to_csr import DeleteAllStates, RemoveAcceleratorOps
from compiler.accelerators.util import find_accelerator_op

@dataclass
class LowerAccfgBasePattern(RewritePattern, ABC):
    """
    Base class for the accfg dialect lowerings.

    Wraps some common logic to get handles to accelerator ops inside the module.
    """

    module: builtin.ModuleOp

    @cache
    def get_acc(
        self, accelerator: StringAttr
    ) -> tuple[accfg.AcceleratorOp, type[Accelerator]]:
        fake_registry = {
                "gemmini_ex" : GemminiExAccelerator(),
                "gemmini_mvin" : GemminiMvinAccelerator(),
                "gemmini_mvout" : GemminiMvoutAccelerator(),
        }
        acc_op = find_accelerator_op(self.module, accelerator)

        if not acc_op:
            raise RuntimeError(
                f"Symbol Table lookup failed for accelerator '{accelerator.data}'. "
                "Is the symbol declared by an accfg.accelerator op in the module?"
            )
        else:
            acc_name = acc_op.name_prop.string_value()
            return acc_op, fake_registry[acc_name]

    def __hash__(self):
        return id(self)


class LowerAccfgSetupToCsr(LowerAccfgBasePattern):
    """
    Convert setup ops to a series of CSR sets that set each field to the given value.

    Looks up the csr addresses of the value fields by getting the `accfg.accelerator`
    operation from the module op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.SetupOp, rewriter: PatternRewriter, /):
        acc_op, acc_info = self.get_acc(op.accelerator)
        # grab a dict that translates field names to CSR addresses:
        # emit the llvm assembly code to set csr values:
        rewriter.replace_matched_op(
            acc_info.lower_acc_setup(op, acc_op),
            [None],
            safe_erase=False,
        )


class LowerAccfgLaunchToCsr(LowerAccfgBasePattern):
    """
    Convert launch ops to a single `csr_set $launch_addr, 1`
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.LaunchOp, rewriter: PatternRewriter, /):
        assert isinstance(op.state.type, accfg.StateType)
        acc_op, acc_info = self.get_acc(op.state.type.accelerator)

        # insert an op that sets the launch CSR to 1
        rewriter.replace_matched_op(
            acc_info.lower_acc_launch(op, acc_op),
            [op.state],
            safe_erase=False,
        )


class LowerAccfgAwaitToCsr(LowerAccfgBasePattern):
    """
    Lower await ops to a set of assembly that lowers to a buffer.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: accfg.AwaitOp, rewriter: PatternRewriter, /):
        assert isinstance(op.token.type, accfg.StateType | accfg.TokenType)
        acc_op, acc_info = self.get_acc(op.token.type.accelerator)

        # emit a snax_hwpe-style barrier
        rewriter.replace_matched_op(
            acc_info.lower_acc_await(acc_op),
            safe_erase=False,
        )



class ConvertGemminiOsToRocc(ModulePass):
    """
    Converts accfg dialect ops to series of SNAX-like csr sets.
    """

    name = "convert-gemmini-os-to-rocc"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # first lower all accfg ops and erase old SSA values
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAccfgSetupToCsr(op),
                    LowerAccfgLaunchToCsr(op),
                    LowerAccfgAwaitToCsr(op),
                ]
            ),
            walk_reverse=True,
        ).rewrite_module(op)

        # then we remove all the top-level accfg.accelerator operations from the module and erase the state variables
        PatternRewriteWalker(
            GreedyRewritePatternApplier([DeleteAllStates(), RemoveAcceleratorOps()])
        ).rewrite_module(op)
