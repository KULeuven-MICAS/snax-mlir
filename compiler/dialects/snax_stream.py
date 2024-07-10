from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    NoTerminator,
    ParametrizedAttribute,
    Region,
    SSAValue,
)
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    ParameterDef,
    irdl_attr_definition,
    prop_def,
    region_def,
    var_operand_def,
)
from xdsl.irdl.irdl import irdl_op_definition
from xdsl.parser import AttrParser
from xdsl.printer import Printer


@irdl_attr_definition
class StridePattern(ParametrizedAttribute):
    """
    Attribute representing a strided streaming pattern in SNAX.
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

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = [")
            printer.print_list(self.upper_bounds, lambda attr: printer.print(attr.data))
            printer.print_string("], ts = [")
            printer.print_list(
                self.temporal_strides, lambda attr: printer.print(attr.data)
            )
            printer.print_string("], ss = [")
            printer.print_list(
                self.spatial_strides, lambda attr: printer.print(attr.data)
            )
            printer.print_string("]")

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> Sequence[Attribute]:
        with parser.in_angle_brackets():
            parser.parse_identifier("ub")
            parser.parse_punctuation("=")
            ub = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("ts")
            parser.parse_punctuation("=")
            ts = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            parser.parse_punctuation(",")
            parser.parse_identifier("ss")
            parser.parse_punctuation("=")
            ss = ArrayAttr(
                IntAttr(i)
                for i in parser.parse_comma_separated_list(
                    parser.Delimiter.SQUARE, parser.parse_integer
                )
            )
            return (ub, ts, ss)


@irdl_op_definition
class StreamingRegionOp(IRDLOperation):
    name = "snax_stream.streaming_region"

    # inputs and outputs are pointers to the base address of the data in memory
    inputs = var_operand_def(IndexType)
    outputs = var_operand_def(IndexType)

    # streaming stride pattern
    # there should be one stride pattern for every input/output
    # the upper bounds of all stride patterns should be equal
    stride_pattern = prop_def(ArrayAttr[StridePattern])

    body = region_def("single_block")

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    traits = frozenset((NoTerminator(),))

    def __init__(
        self,
        inputs: Sequence[SSAValue],
        outputs: Sequence[SSAValue],
        stride_patterns: ArrayAttr[StridePattern] | Sequence[StridePattern],
        body: Region,
    ) -> None:
        if not isinstance(stride_patterns, ArrayAttr):
            stride_patterns = ArrayAttr(stride_patterns)
        super().__init__(
            operands=[inputs, outputs],
            regions=[body],
            properties={"stride_patterns": stride_patterns},
        )


SnaxStream = Dialect("snax_stream", [StreamingRegionOp], [StridePattern])
