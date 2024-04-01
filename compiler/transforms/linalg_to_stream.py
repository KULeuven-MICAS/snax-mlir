from xdsl.dialects import builtin
from xdsl.dialects.linalg import Generic
from xdsl.ir import MLContext
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.util.kernel_type import KernelType


class LinalgToConstantOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, generic_op: Generic, rewriter: PatternRewriter):
        print("found a linalg generic op!")

        kernel_type = KernelType.get_kernel(generic_op)

        if not kernel_type:
            return

        print(f"detected a kernel type of {kernel_type}")

        # zigzag -> 3 operands: 2 inputs, 1 output
        assert len(generic_op.outputs) == 1
        generic_op.outputs[0]

        assert len(generic_op.inputs) == 2
        generic_op.inputs[0]
        generic_op.inputs[1]

        zigzag_description = dict()

        zigzag_description["operator_type"] = "default"

        # construct equation
        output_access = "O"
        for i in range(len(generic_op.indexing_maps.data[-1].data.results)):
            map = generic_op.indexing_maps.data[-1].data.results[i]
            assert isinstance(map, AffineDimExpr)
            output_access += f"[{str(map)}]"

        input_a_access = "A"
        for i in range(len(generic_op.indexing_maps.data[0].data.results)):
            map = generic_op.indexing_maps.data[0].data.results[i]
            assert isinstance(map, AffineDimExpr)
            input_a_access += f"[{str(map)}]"

        input_b_access = "B"
        for i in range(len(generic_op.indexing_maps.data[1].data.results)):
            map = generic_op.indexing_maps.data[1].data.results[i]
            assert isinstance(map, AffineDimExpr)
            input_b_access += f"[{str(map)}]"

        if kernel_type == KernelType.MUL:
            zigzag_description[
                "equation"
            ] = f"{output_access} = {input_a_access} * {input_b_access}"

        elif kernel_type in (KernelType.MAC, KernelType.QMAC):
            zigzag_description[
                "equation"
            ] = f"{output_access} += {input_a_access} * {input_b_access}"

        zigzag_description["dimension_relations"] = []

        results = []
        results.append(generic_op.indexing_maps.data[0].data.results)
        results.append(generic_op.indexing_maps.data[1].data.results)
        results.append(generic_op.indexing_maps.data[2].data.results)

        combined_affine_map = AffineMap(3, 0, results)

        combined_affine_map.inverse_permutation()

        pass


class LinalgToStream(ModulePass):
    name = "linalg-to-stream"

    def apply(self, ctx: MLContext, module: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            LinalgToConstantOp(), apply_recursively=False
        ).rewrite_module(module)
