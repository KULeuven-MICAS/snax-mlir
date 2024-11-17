from collections.abc import Iterable, Sequence
from typing import Generic, TypeVar

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyShapedType,
    ArrayAttr,
    ContainerType,
    ShapedType,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    IsTerminator,
    NoTerminator,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    var_operand_def,
    var_result_def,
)


@irdl_op_definition
class PipelineOp(IRDLOperation):
    name = "pipeline.pipeline"

    body = region_def("single_block")

    traits = frozenset([NoTerminator()])

    def __init__(self, body: Region) -> None:
        super().__init__(regions=[body])


@irdl_op_definition
class StageOp(IRDLOperation):
    name = "pipeline.stage"

    input_buffers = var_operand_def()
    output_buffers = var_operand_def()

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset([NoTerminator()])

    def __init__(
        self,
        input_buffers: Sequence[SSAValue | Operation],
        output_buffers: Sequence[SSAValue | Operation],
        body: Region,
    ) -> None:
        # input and output buffers should be alloc operations
        super().__init__(operands=[input_buffers, output_buffers], regions=[body])


@irdl_op_definition
class DoubleStageOp(IRDLOperation):
    name = "pipeline.double_stage"

    input_buffers_even = var_operand_def()
    input_buffers_odd = var_operand_def()

    output_buffers_even = var_operand_def()
    output_buffers_odd = var_operand_def()

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset([NoTerminator()])

    def __init__(
        self,
        input_buffers_even: Sequence[SSAValue | Operation],
        input_buffers_odd: Sequence[SSAValue | Operation],
        output_buffers_even: Sequence[SSAValue | Operation],
        output_buffers_odd: Sequence[SSAValue | Operation],
        body: Region,
    ) -> None:
        # input and output buffers should be alloc operations
        super().__init__(
            operands=[input_buffers_even, input_buffers_odd, output_buffers_even, output_buffers_odd], regions=[body]
        )


Pipeline = Dialect(
    "pipeline",
    [
        # ops
        PipelineOp,
        StageOp,
        DoubleStageOp,
    ],
    [
        # attributes
    ],
)
