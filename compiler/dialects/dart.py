from collections.abc import Iterable, Mapping, Sequence
from typing import Generic, TypeVar

from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    ContainerType,
    IndexType,
    IntegerAttr,
    ShapedType,
    StringAttr,
)
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import (
    Attribute,
    Dialect,
    IsTerminator,
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
    traits_def,
    var_operand_def,
    var_result_def,
)

"""
Custom `dart` dialect, heavily inspired by xDSL `stream` dialect, to simplify
things in a (hopefully) more principled approach, including:
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

    name = "dart.stream"

    element_type: ParameterDef[_StreamTypeElement]

    def __init__(self, element_type: _StreamTypeElement):
        super().__init__([element_type])

    def get_element_type(self) -> _StreamTypeElement:
        return self.element_type


class StreamingRegionOpBase(IRDLOperation):
    """
    An operation that creates streams from tensors or memrefs, which are only available to
    read from within the body of the operation.

    Within the loop body, memrefs/tensors that are streamed must not be otherwise accessed
    via any other access means, including extraction (e.g.: memref.view).
    """

    inputs = var_operand_def()
    outputs = var_operand_def()
    result_tensors = var_result_def()
    patterns = prop_def(ArrayAttr[AffineMapAttr])

    body = region_def("single_block")

    accelerator = opt_prop_def(StringAttr)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        outputs: Sequence[SSAValue | Operation],
        patterns: ArrayAttr[AffineMapAttr],
        body: Region,
        accelerator: str | StringAttr | None = None,
        result_types: Sequence[Attribute] = (),
        other_props: Mapping[str, Attribute | None] = {},
    ) -> None:
        if isinstance(accelerator, str):
            accelerator = StringAttr(accelerator)
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={
                "patterns": patterns,
                "accelerator": accelerator,
                **other_props,
            },
            result_types=[result_types],
        )


@irdl_op_definition
class OperationOp(StreamingRegionOpBase):
    """
    A streaming region op that represents an unscheduled operation,
    with streams mapping the iteration space to the operand indexing space.
    """

    name = "dart.operation"

    def get_pattern_bounds_to_shapes_map(self) -> AffineMap:
        """
        Returns mapping from pattern iteration bounds to operand shapes
        """
        return AffineMap(
            self.patterns.data[0].data.num_dims,
            self.patterns.data[0].data.num_symbols,
            tuple(res for map in self.patterns for res in map.data.results),
        )

    def get_shapes_to_pattern_bounds_map(self) -> AffineMap:
        """
        Returns mapping from operand shapes to pattern iteration bounds.
        """
        assert (result := self.get_pattern_bounds_to_shapes_map().inverse_permutation())
        return result

    def get_static_shapes(self) -> Iterable[int]:
        """
        Return the static shapes of all operands of this op.
        """
        for operand in self.operands:
            assert isinstance(operand.type, ShapedType)
            yield from operand.type.get_shape()

    def get_static_pattern_bounds(self) -> Iterable[int]:
        """
        Return the static pattern bounds of this op.
        """
        return self.get_shapes_to_pattern_bounds_map().eval(
            tuple(self.get_static_shapes()), []
        )


@irdl_op_definition
class ScheduleOp(StreamingRegionOpBase):
    """
    A streaming region op that represents an scheduled operation,
    with streams mapping the iteration space to the operand indexing space.

    Due to the transformations that took place on the unscheduled op,
    some extra metadata is needed, consisting of the fixed bounds
    of the iteration space and the tile sizes that are used if a tiling
    transformation took place.
    """

    name = "dart.schedule"

    # The bounds of the iteration space of the schedule
    bounds = prop_def(ParameterDef[ArrayAttr[IntegerAttr[IndexType]]])

    # The tiling factors for the different dimensions of inputs and outputs
    tiles = prop_def(ParameterDef[ArrayAttr[ArrayAttr[IntegerAttr[IndexType]]]])

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        outputs: Sequence[SSAValue | Operation],
        patterns: ArrayAttr[AffineMapAttr],
        body: Region,
        bounds: ArrayAttr[IntegerAttr[IndexType]] | Sequence[int],
        tiles: ArrayAttr[ArrayAttr[IntegerAttr[IndexType]]] | Sequence[Sequence[int]],
        accelerator: str | StringAttr | None = None,
        result_types: Sequence[Attribute] = (),
    ) -> None:
        if isinstance(tiles, Sequence):
            tiles = ArrayAttr(
                [
                    ArrayAttr([IntegerAttr.from_index_int_value(val) for val in tile])
                    for tile in tiles
                ]
            )
        if isinstance(bounds, Sequence):
            bounds = ArrayAttr(
                [IntegerAttr.from_index_int_value(val) for val in bounds]
            )
        super().__init__(
            inputs,
            outputs,
            patterns,
            body,
            accelerator,
            result_types,
            {"tiles": tiles, "bounds": bounds},
        )


@irdl_op_definition
class AccessPatternOp(StreamingRegionOpBase):
    """
    A streaming region op that represents an scheduled operation, after
    layout resolution, with streams mapping the iteration space to memory.
    """

    name = "dart.access_pattern"

    # The bounds of the iteration space of the schedule
    bounds = prop_def(ParameterDef[ArrayAttr[IntegerAttr[IndexType]]])

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        outputs: Sequence[SSAValue | Operation],
        patterns: ArrayAttr[AffineMapAttr],
        body: Region,
        bounds: ArrayAttr[IntegerAttr[IndexType]] | Sequence[int],
        accelerator: str | StringAttr | None = None,
        result_types: Sequence[Attribute] = (),
    ) -> None:
        if isinstance(bounds, Sequence):
            bounds = ArrayAttr(
                [IntegerAttr.from_index_int_value(val) for val in bounds]
            )
        super().__init__(
            inputs,
            outputs,
            patterns,
            body,
            accelerator,
            result_types,
            {"bounds": bounds},
        )


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "dart.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class GenericOp(IRDLOperation):
    """
    Generic that operates on streams, similar to a linalg.generic.
    Indexing maps / iterators are not relevant, so they are not included.
    """

    name = "dart.generic"

    # inputs can be streams or integers
    inputs = var_operand_def()
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


Dart = Dialect(
    "dart",
    [
        OperationOp,
        ScheduleOp,
        AccessPatternOp,
        GenericOp,
        YieldOp,
    ],
    [
        StreamType,
    ],
)
