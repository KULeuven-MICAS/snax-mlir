from __future__ import annotations

from collections.abc import Sequence


from xdsl.dialects.builtin import (
    AnyShapedType,
    AnyTensorType,
    ShapedType,
    StringAttr,
)
from xdsl.ir import Dialect, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    VarOperand,
    VarOpResult,
    irdl_op_definition,
    opt_attr_def,
    region_def,
    var_operand_def,
    var_result_def,
)


@irdl_op_definition
class Mul(IRDLOperation):
    name = "linalg.mul"

    inputs: VarOperand = var_operand_def()
    outputs: VarOperand = var_operand_def(AnyShapedType())

    res: VarOpResult = var_result_def(AnyTensorType)

    body: Region = region_def("single_block")

    # Trait attributes
    library_call: StringAttr | None = opt_attr_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        body: Region,
        library_call: StringAttr | None = None,
    ) -> None:
        super().__init__(
            operands=[inputs, outputs],
            result_types=[[]],
            attributes={
                "library_call": library_call,
            },
            regions=[body],
        )

    def get_static_shapes(self) -> list[int]:
        sizes: list[int] = []
        for input in self.inputs:
            if isinstance(input.type, ShapedType):
                for dim in input.type.get_shape():
                    sizes.append(dim)
        for output in self.outputs:
            if isinstance(output.type, ShapedType):
                for dim in output.type.get_shape():
                    sizes.append(dim)
        return sizes


# Extended Linalg Dialect
LinalgExtension = Dialect("linalg_extension", [Mul])
