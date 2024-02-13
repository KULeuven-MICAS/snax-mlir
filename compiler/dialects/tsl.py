from __future__ import annotations

from math import prod

from xdsl.dialects.arith import Constant, DivUI, Muli
from xdsl.dialects.builtin import IndexType
from xdsl.dialects.memref import Dim
from xdsl.ir import Data, Dialect, Operation, SSAValue
from xdsl.irdl import (
    irdl_attr_definition,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer

from compiler.ir.tsl import TiledStridedLayout
from compiler.parser.tsl_parser import TSLParser


@irdl_attr_definition
class TiledStridedLayoutAttr(Data[TiledStridedLayout]):
    """An Attribute containing an TiledStridedLayout object."""

    name = "tsl.tsl"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> TiledStridedLayout:
        with parser.in_angle_brackets():
            tslparser = TSLParser(parser._parser_state)
            return tslparser.parse()

    def print_parameter(self, printer: Printer) -> None:
        printer.print_string(f"<{self.data}>")

    def get_bound_ops(
        self, memref: SSAValue | Operation
    ) -> tuple[list[Operation], dict[tuple[int, int], Operation]]:
        """Generate ops to get the bounds of the Strides in the TSL
        The function handles dynamic strides as well

        Args:
            memref (SSAValue | Operation): The memref to which this
            TSL is applied.

        Returns:
            Result (List[Operation]): the list of operations that must be inserted
            in the ir in the correct ordering.

            Result_mapping (Dict[(int, int), Operation]): a mapping from the tuple
            (dim, depth) to the operation of the bound of the stride at that dim
            and depth. This is used to keep track of which sequence of operations
            was made for which TSL Stride.
        """
        result: list[Operation] = []
        result_mapping: dict[(int, int), Operation] = {}

        tsl = self.data

        for dim in range(tsl.dimension()):
            for depth in range(tsl.tstrides[dim].depth()):
                stride = tsl.get_stride(dim, depth)

                # static case
                if stride.bound is not None:
                    bound_op = Constant.from_int_and_width(stride.bound, IndexType())
                    result.append(bound_op)
                    result_mapping[(dim, depth)] = bound_op

                # dynamic case
                # to calculate the bound of a dynamic stride,
                # we must divide the size of the memref by the product
                # of all lower tile sizes
                else:
                    # get the size of the memref
                    dim_index_op = Constant.from_int_and_width(dim, IndexType())
                    dim_op = Dim.from_source_and_index(memref, dim_index_op)

                    # get the product of all lower tile sizes
                    product_tilebounds = prod(
                        [
                            stride.bound
                            for _, stride in tsl.tstrides[dim]
                            if stride.bound
                        ]
                    )
                    div_op = Constant.from_int_and_width(
                        product_tilebounds, IndexType()
                    )

                    # divide the size of the memref by the product of all lower tiles
                    bound_op = DivUI(dim_op.result, div_op.result, IndexType())

                    # add the ops to result
                    result.extend([dim_index_op, dim_op, div_op, bound_op])
                    result_mapping[(dim, depth)] = bound_op

        return result, result_mapping

    def get_step_ops(
        self, bound_ops: dict[(int, int), Operation]
    ) -> (list[Operation], dict[(int, int), Operation]):
        """Generate ops to get the steps of the Strides in the TSL
        The function handles dynamic strides as well

        Args:
            bound_ops (Dict[(int, int), Operation]): The bound ops of the given
            TSL attribute. These are necessary to calculate the step of a dynamic
            Stride. The argument is a mapping of (dim, depth) to the operation
            of the bound of the stride at that dim and depth.

        Returns:
            Result (List[Operation]): the list of operations that must be inserted
            in the ir in the correct ordering.

            Result_mapping (Dict[(int, int), Operation]): a mapping from the tuple
            (dim, depth) to the operation of the step of the stride at that dim
            and depth
        """
        result: list[Operation] = []
        result_mapping: dict[(int, int), Operation] = {}

        tsl = self.data

        # to handle the dynamic case, we must first find the largest
        # statically defined step, and then use that to calculate the
        # dynamic steps
        max_key = None
        max_value = 0
        for dim, depth, stride in self.data:
            if stride.step and stride.step > max_value:
                max_key = (dim, depth)
                max_value = stride.step

        # generate ops for the maximum
        # the max static stride multiplied by the bound of that Stride
        # can be used as a starting value for the dynamic strides
        max_stride_op = Constant.from_int_and_width(max_value, IndexType())
        dynamic_step = Muli(
            bound_ops[max_key],
            max_stride_op,
            IndexType(),
        )
        result.append(max_stride_op)

        # assign strides right to left
        for dim in reversed(range(tsl.dimension())):
            # assign strides from innermost to outermost
            for depth in reversed(range(tsl.tstrides[dim].depth())):
                stride = tsl.get_stride(dim, depth)

                # static case
                if stride.step is not None:
                    step_op = Constant.from_int_and_width(stride.step, IndexType())
                    result.append(step_op)
                    result_mapping[(dim, depth)] = step_op

                # dynamic case
                else:
                    step_op = dynamic_step
                    dynamic_step = Muli(step_op, bound_ops[(dim, depth)], IndexType())
                    result.append(step_op)
                    result_mapping[(dim, depth)] = step_op

        return result, result_mapping


TSL = Dialect("tsl", [], [TiledStridedLayoutAttr])
