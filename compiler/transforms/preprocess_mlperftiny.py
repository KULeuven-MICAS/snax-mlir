from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects import builtin, func, memref, tensor
from xdsl.ir import BlockArgument
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import SymbolTable
from xdsl.transforms.mlir_opt import MLIROptPass


class InsertStaticFunctionCall(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: builtin.ModuleOp, rewriter: PatternRewriter):
        """
        Insert a function with static arguments them to dynamic ones,
        calls the existing main function, and then casts the dyanmic output
        to a static value. After inlining, everything is static then.
        """

        # define types
        static_input = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_input = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        static_output = builtin.TensorType(builtin.IntegerType(8), (8, 640))
        dynamic_output = builtin.TensorType(builtin.IntegerType(8), (-1, 640))

        @Builder.implicit_region((static_input,))
        def func_region(args: tuple[BlockArgument, ...]):
            cast = tensor.CastOp(args[0], dynamic_input)
            func_result = func.Call("main", [cast], [dynamic_output])
            cast2 = tensor.CastOp(func_result, static_output)
            func.Return(cast2)

        main_func = func.FuncOp(
            "run_network", builtin.FunctionType.from_lists([static_input], [static_output]), region=func_region
        )

        rewriter.insert_op(main_func, InsertPoint.at_end(op.body.block))


class DropOldFunction(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        ## delete old main function which has been inlined
        if op.sym_name.data == "main":
            rewriter.erase_matched_op()


class AllocToGlobal(RewritePattern):
    index: int = 0
    """
    Convert all static allocs to empty statically scheduled memref globals
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Alloc, rewriter: PatternRewriter):
        # get module op
        module_op = op
        while not isinstance(module_op, builtin.ModuleOp):
            assert module_op is not None
            module_op = module_op.parent_op()

        symbol_table = module_op.get_trait(SymbolTable)
        assert symbol_table

        # create global symbol name
        global_sym_name = "_static_const_" + str(self.index)
        self.index = self.index + 1

        # check if it is not in the symbol table
        assert SymbolTable.lookup_symbol(module_op, global_sym_name) is None

        # create global
        memref_global = memref.Global.get(
            builtin.StringAttr(global_sym_name), op.results[0].type, initial_value=builtin.UnitAttr()
        )
        SymbolTable.insert_or_update(module_op, memref_global)

        # remove all the deallocs
        deallocs = [
            user_op.operation for user_op in op.results[0].uses if isinstance(user_op.operation, memref.Dealloc)
        ]
        for dealloc in deallocs:
            rewriter.erase_op(dealloc)

        # replace op by get global
        memref_get = memref.GetGlobal(global_sym_name, op.results[0].type)

        rewriter.replace_matched_op(memref_get)


@dataclass(frozen=True)
class PreprocessMLPerfTiny(ModulePass):
    name = "preprocess-mlperftiny"

    executable: str = field(default="mlir-opt")
    generic: bool = field(default=True)

    mlir_inliner_pass = MLIROptPass(
        arguments=("-inline", "-cse", "-canonicalize", "-mlir-print-local-scope", "-mlir-print-op-generic")
    )
    mlir_bufferization_pass = MLIROptPass(
        arguments=(
            '--test-linalg-transform-patterns="test-generalize-pad-tensor"',
            "--linalg-generalize-named-ops",
            "--empty-tensor-to-alloc-tensor",
            '--one-shot-bufferize="bufferize-function-boundaries allow-return-allocs'
            + ' function-boundary-type-conversion=identity-layout-map"',
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
        )
    )

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        # self.mlir_opt_pass.apply(ctx, op)
        # PatternRewriteWalker(InsertStaticFunctionCall(), apply_recursively=False).rewrite_module(op)
        # self.mlir_inliner_pass.apply(ctx, op)
        # PatternRewriteWalker(DropOldFunction()).rewrite_module(op)
        # self.mlir_bufferization_pass.apply(ctx, op)
        PatternRewriteWalker(AllocToGlobal()).rewrite_module(op)
