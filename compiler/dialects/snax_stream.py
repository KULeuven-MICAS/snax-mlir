from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr, IndexType, IntAttr, ModuleOp, StringAttr
from xdsl.ir import (
    Attribute,
    Dialect,
    NoTerminator,
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
    var_operand_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from compiler.accelerators.streamers.streamers import StreamerConfiguration


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

    def print_parameters(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string("ub = [")
            printer.print_list(self.upper_bounds, lambda attr: printer.print(attr.data))
            printer.print_string("], ts = [")
            printer.print_list(self.temporal_strides, lambda attr: printer.print(attr.data))
            printer.print_string("], ss = [")
            printer.print_list(self.spatial_strides, lambda attr: printer.print(attr.data))
            printer.print_string("]")

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
    # the upper bounds of all stride patterns should be equal
    stride_pattern = prop_def(ArrayAttr[StridePattern])

    accelerator = prop_def(StringAttr)
    """
    Name of the accelerator this region is for
    """

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

    def verify_(self):
        # import only here to avoid circular import errors
        from compiler.accelerators.registry import AcceleratorRegistry
        from compiler.dialects.snax import StreamerConfigurationAttr

        module_op = self
        while module_op and not isinstance(module_op, ModuleOp):
            module_op = module_op.parent_op()
        if not module_op:
            raise VerifyException("ModuleOp not found!")
        accelerator_op, _ = AcceleratorRegistry().lookup_acc_info(self.accelerator, module_op)

        if not accelerator_op:
            raise VerifyException("AcceleratorOp for streaming_region not found!")

        streamer_interface = accelerator_op.get_attr_or_prop("streamer_config")

        if not streamer_interface or not isinstance(streamer_interface, StreamerConfigurationAttr):
            raise VerifyException("Specified accelerator does not implement the streamer interface")

        streamer_config: StreamerConfiguration = streamer_interface.data

        if len(self.stride_pattern) != streamer_config.size():
            raise VerifyException("Number of streamers does not equal number of stride patterns")

        for stride_pattern, streamer in zip(self.stride_pattern, streamer_config.streamers):

            if len(stride_pattern.temporal_strides) > streamer.temporal_dim:
                raise VerifyException("Temporal stride pattern exceeds streamer dimensionality")

            if len(stride_pattern.spatial_strides) > streamer.spatial_dim:
                raise VerifyException("Spatial stride pattern exceeds streamer dimensionality")


SnaxStream = Dialect("snax_stream", [StreamingRegionOp], [StridePattern])
