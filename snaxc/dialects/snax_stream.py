from collections.abc import Sequence
from typing import Self

from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, StringAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    NoTerminator,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    VerifyException,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    region_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from snaxc.accelerators import find_accelerator_op
from snaxc.accelerators.streamers.streamers import StreamerConfiguration
from snaxc.dialects.snax import StreamerConfigurationAttr


@irdl_attr_definition
class StridePattern(ParametrizedAttribute):
    """
    Attribute representing a strided streaming pattern in SNAX.
    A stride pattern coincides with the Data Streamers from the SNAX Framework.
    For detailed information, see: https://github.com/KULeuven-MICAS/snax_cluster/blob/main/hw/chisel/doc/streamer.md
    The upper bounds and temporal strides define a set of nested for loops
    where data is accessed sequentially in a structured pattern. Every execution,
    a number of data elements is acccessed in parallel. The distance between these
    elements is denoted by the spatial strides.
    """

    name = "snax_stream.stride_pattern"

    upper_bounds: ParameterDef[ArrayAttr[IntAttr]]
    temporal_strides: ParameterDef[ArrayAttr[IntAttr]]
    spatial_strides: ParameterDef[ArrayAttr[IntAttr]]

    def __init__(
        self,
        upper_bounds: ParameterDef[ArrayAttr[IntAttr]] | Sequence[int],
        temporal_strides: ParameterDef[ArrayAttr[IntAttr]] | Sequence[int],
        spatial_strides: ParameterDef[ArrayAttr[IntAttr]] | Sequence[int],
    ):
        parameters: Sequence[Attribute] = []
        for arg in (upper_bounds, temporal_strides, spatial_strides):
            if not isinstance(arg, ArrayAttr):
                arg = ArrayAttr([IntAttr(x) for x in arg])
            parameters.append(arg)
        super().__init__(parameters)

    def verify(self):
        if len(self.upper_bounds) != len(self.temporal_strides):
            raise VerifyException("Number of upper bounds should be equal to number of strides")

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = [")
            printer.print_list(self.upper_bounds, lambda attr: printer.print(attr.data))
            printer.print_string("], ts = [")
            printer.print_list(self.temporal_strides, lambda attr: printer.print(attr.data))
            printer.print_string("], ss = [")
            printer.print_list(self.spatial_strides, lambda attr: printer.print(attr.data))
            printer.print_string("]")

    def canonicalize(self) -> Self:
        if IntAttr(0) in self.spatial_strides.data:
            return self
        upper_bounds = [x.data for x in self.upper_bounds.data]
        temporal_strides = [x.data for x in self.temporal_strides.data]

        new_upper_bounds: list[int] = []
        new_temporal_strides: list[int] = []

        for ub, ts in zip(upper_bounds, temporal_strides):
            # hardcoded zeros should pass through for disabled streamers
            if ub == 0:
                new_upper_bounds.append(0)
                new_temporal_strides.append(0)
            # upper bound of 1 can be removed
            elif ub == 1:
                pass
            elif len(new_upper_bounds) and new_upper_bounds[-1] * new_temporal_strides[-1] == ts:
                # wrap two strides in 1
                new_upper_bounds[-1] *= ub
            else:
                new_upper_bounds.append(ub)
                new_temporal_strides.append(ts)

        return type(self)(new_upper_bounds, new_temporal_strides, self.spatial_strides)

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_identifier("ub")
            parser.parse_punctuation("=")
            ub = ArrayAttr(
                IntAttr(i) for i in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer)
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("ts")
            parser.parse_punctuation("=")
            ts = ArrayAttr(
                IntAttr(i) for i in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer)
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("ss")
            parser.parse_punctuation("=")
            ss = ArrayAttr(
                IntAttr(i) for i in parser.parse_comma_separated_list(parser.Delimiter.SQUARE, parser.parse_integer)
            )
            return (ub, ts, ss)


@irdl_op_definition
class StreamingRegionOp(IRDLOperation):
    """
    An operation that creates streams from access patterns realized by SNAX Streamers.
    These streams are accessable only to ops within the body of the operation.
    """

    name = "snax_stream.streaming_region"

    # inputs and outputs are pointers to the base address of the data in memory
    inputs = var_operand_def(IndexType)
    outputs = var_operand_def(IndexType)

    # streaming stride pattern
    # there should be one stride pattern for every input/output
    stride_patterns = prop_def(ArrayAttr[StridePattern])

    accelerator = prop_def(StringAttr)
    """
    Name of the accelerator this region is for
    """

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = traits_def(NoTerminator())

    def __init__(
        self,
        inputs: Sequence[SSAValue | Operation],
        outputs: Sequence[SSAValue | Operation],
        stride_patterns: ArrayAttr[StridePattern] | Sequence[StridePattern],
        accelerator: StringAttr | str,
        body: Region,
    ) -> None:
        if not isinstance(stride_patterns, ArrayAttr):
            stride_patterns = ArrayAttr(stride_patterns)
        if isinstance(accelerator, str):
            accelerator = StringAttr(accelerator)
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={"stride_patterns": stride_patterns, "accelerator": accelerator},
        )

    def verify_(self):
        acc_op = find_accelerator_op(self, self.accelerator.data)
        if not acc_op:
            raise VerifyException("AcceleratorOp not found!")

        streamer_interface = acc_op.get_attr_or_prop("streamer_config")
        if not streamer_interface or not isinstance(streamer_interface, StreamerConfigurationAttr):
            raise VerifyException("Specified accelerator does not contain a StreamerConfigurationAttr")

        streamer_config: StreamerConfiguration = streamer_interface.data

        if len(self.stride_patterns) != streamer_config.size():
            raise VerifyException("Number of streamers does not equal number of stride patterns")

        for stride_pattern, streamer in zip(self.stride_patterns, streamer_config.streamers):
            if len(stride_pattern.temporal_strides) > streamer.temporal_dim:
                raise VerifyException("Temporal stride pattern exceeds streamer dimensionality")

            if len(stride_pattern.spatial_strides) > streamer.spatial_dim:
                raise VerifyException("Spatial stride pattern exceeds streamer dimensionality")


SnaxStream = Dialect("snax_stream", [StreamingRegionOp], [StridePattern])
