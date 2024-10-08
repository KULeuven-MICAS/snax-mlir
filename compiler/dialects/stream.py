from collections.abc import Sequence
from typing import Generic, TypeVar

from xdsl.dialects.builtin import (
    AffineMapAttr,
    AnyShapedType,
    ArrayAttr,
    ContainerType,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    IsTerminator,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
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

"""
Custom `stream` dialect, to simplify things in a more principled approach, including:
- inherent support for tensors
- streams with value semantics
- no specified static bounds in access patterns: they are just affine maps
- no stream ops outside of streaming regions allowed
"""

_StreamTypeElement = TypeVar("_StreamTypeElement", bound=Attribute)


@irdl_attr_definition
class StreamType(
    Generic[_StreamTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StreamTypeElement],
):
    """
    A stream type with value semantics.
    A stream is defined by an element type, and is produced as the result of
    an operation, or through a streaming region op.
    Streams can only be read from, there is no distinction between readable/writeable streams.
    """

    name = "stream.stream"

    element_type: ParameterDef[_StreamTypeElement]

    def __init__(self, element_type: _StreamTypeElement):
        super().__init__([element_type])

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type


@irdl_op_definition
class StreamingRegionOp(IRDLOperation):
    """
    An operation that creates streams from tensors or memrefs, which are only available to
    read from within the body of the operation.

    Within the loop body, memrefs/tensors that are streamed must not be otherwise accessed
    via any other access means, including extraction (e.g.: memref.view).
    """

    name = "stream.streaming_region"

    inputs = var_operand_def(AnyShapedType())
    outputs = var_operand_def(AnyShapedType())
    result_tensors = var_result_def()
    patterns = prop_def(ArrayAttr[AffineMapAttr])

    body = region_def("single_block")

    accelerator = opt_prop_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        patterns: ArrayAttr[AffineMapAttr],
        body: Region,
        accelerator: str | StringAttr | None = None,
        result_types: Sequence[Attribute] = (),
    ) -> None:
        if isinstance(accelerator, str):
            accelerator = StringAttr(accelerator)
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={
                "patterns": patterns,
                "accelerator": accelerator,
            },
            result_types=[result_types],
        )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "stream.yield"

    traits = frozenset([IsTerminator()])


@irdl_op_definition
class GenericOp(IRDLOperation):
    """
    Generic that operates on streams, similar to a linalg.generic.
    Indexing maps / iterators are not relevant, so they are not included.
    """

    name = "stream.generic"

    inputs = var_operand_def(StreamType)
    outputs = var_result_def(StreamType)

    body = region_def()

    # Trait attributes
    doc = opt_prop_def(StringAttr)
    library_call = opt_prop_def(StringAttr)

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        body: Region,
        doc: StringAttr | None = None,
        library_call: StringAttr | None = None,
        result_types: Sequence[Attribute] = (),
    ) -> None:
        super().__init__(
            operands=[inputs],
            properties={
                "doc": doc,
                "library_call": library_call,
            },
            regions=[body],
            result_types=[result_types],
        )


Stream = Dialect(
    "stream",
    [
        StreamingRegionOp,
        GenericOp,
        YieldOp,
    ],
    [
        StreamType,
    ],
)

