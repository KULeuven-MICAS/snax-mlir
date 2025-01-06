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

from compiler.accelerators.registry import AcceleratorRegistry
from compiler.accelerators.snax import SNAXStreamer
from compiler.accelerators.util import find_accelerator_op
from compiler.ir.stream.access_pattern import Template

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

    def get_accelerator_info(self) -> Template:
        assert self.accelerator is not None

        # Go and fetch the accelerator op
        accelerator_str = self.accelerator.data
        acc_op = find_accelerator_op(self, accelerator_str)

        if not acc_op:
            raise RuntimeError("AcceleratorOp not found!")

        # get template and template_bounds
        accelerator_type = AcceleratorRegistry().get_acc_info(acc_op)
        assert issubclass(accelerator_type, SNAXStreamer)

        template = accelerator_type.get_template(self)

        return template


@irdl_op_definition
class StreamingRegionOp(StreamingRegionOpBase):
    name = "stream.streaming_region"

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
    name = "stream.schedule"

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
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "stream.yield"

    traits = traits_def(IsTerminator())


@irdl_op_definition
class GenericOp(IRDLOperation):
    """
    Generic that operates on streams, similar to a linalg.generic.
    Indexing maps / iterators are not relevant, so they are not included.
    """

    name = "stream.generic"

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


Stream = Dialect(
    "stream",
    [
        StreamingRegionOp,
        ScheduleOp,
        GenericOp,
        YieldOp,
    ],
    [
        StreamType,
    ],
)
