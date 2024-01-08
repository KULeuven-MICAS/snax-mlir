from __future__ import annotations

from xdsl.parser.base_parser import BaseParser, ParserState
from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Token

from compiler.ir.tsl import Stride, TiledStride, TiledStridedLayout


class TSLParser(BaseParser):
    def __init__(self, state: ParserState) -> None:
        self._resume_from(state)

    def _parse_stride(self) -> list[int]:
        """
        strides ::== `[` stride (`,` stride)* `]`
        """
        self._parse_token(Token.Kind.L_SQUARE, "Expected opening bracket")
        strides: list[int] = []
        while not self._parse_optional_token(Token.Kind.R_SQUARE):
            strides.append(self.parse_integer())
            self._parse_optional_token(Token.Kind.COMMA)
        return strides

    def _parse_bound(self) -> list[int]:
        """
        bounds ::== `[` bound (`,` bound)* `]`
        """
        self._parse_token(Token.Kind.L_SQUARE, "Expected opening bracket")
        bounds: list[int] = []
        while not self._parse_optional_token(Token.Kind.R_SQUARE):
            bounds.append(self.parse_integer())
            self._parse_optional_token(Token.Kind.COMMA)
        return bounds

    def _parse_tiled_stride(self) -> TiledStride:
        """
        tiled-stride ::= strides * bounds
        """
        strides = self._parse_stride()
        self._parse_token(Token.Kind.STAR, "Expected star")
        bounds = self._parse_bound()
        if len(strides) != len(bounds):
            raise ParseError("Expected same number of strides and bounds")
        # construct the tiledstrides
        return TiledStride(
            [Stride(stride, bound) for stride, bound in zip(strides, bounds)]
        )

    def parse(self) -> TiledStridedLayout:
        """
        tsl ::= `(` tiled-stride (`,` tiled-stride)*`, offset: ` offset `)`
        """
        self._parse_token(Token.Kind.L_PAREN, "Expected opening bracket")
        tstrides = []
        self.parse_optional_characters("offset:")
        # while not self._parse_optional_token(Token.Kind.R_PAREN):
        while not self.parse_optional_characters("offset"):
            tstrides.append(self._parse_tiled_stride())
            self._parse_optional_token(Token.Kind.COMMA)
        self._parse_token(Token.Kind.COLON, "Expected colon")
        offset = self.parse_integer()
        self._parse_token(Token.Kind.R_PAREN, "Expected closing bracket")
        return TiledStridedLayout(tstrides, offset=offset)
