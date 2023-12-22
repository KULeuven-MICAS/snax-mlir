from __future__ import annotations

from xdsl.ir import (
    Data,
    Dialect,
)
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


TSL = Dialect("tsl", [], [TiledStridedLayoutAttr])
