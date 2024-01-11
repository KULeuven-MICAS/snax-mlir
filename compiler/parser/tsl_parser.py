from __future__ import annotations

from xdsl.parser.base_parser import BaseParser, ParserState
from xdsl.utils.exceptions import ParseError
from xdsl.utils.lexer import Token

from compiler.ir.tsl import Stride, TiledStride, TiledStridedLayout


class TSLParser(BaseParser):
    def __init__(self, state: ParserState) -> None:
        self._resume_from(state)

    def _parse_int_or_question(self, context_msg: str = "") -> int | None:
        """Parse either an integer literal, or a '?'."""
        if self._parse_optional_token(Token.Kind.QUESTION) is not None:
            return None
        if (v := self.parse_optional_integer(allow_boolean=False)) is not None:
            return v
        self.raise_error("Expected an integer literal or `?`" + context_msg)

    def _parse_stride(self) -> list[int]:
        """
        strides ::== `(` stride (`,` stride)* `)`
        """
        self._parse_token(Token.Kind.L_PAREN, "Expected opening bracket")
        strides: list[int] = []
        while not self._parse_optional_token(Token.Kind.R_PAREN):
            strides.append(self._parse_int_or_question())
            self._parse_optional_token(Token.Kind.COMMA)
        return strides

    def _parse_bound(self) -> list[int]:
        """
        bounds ::== `[` bound (`,` bound)* `]`
        """
        self._parse_token(Token.Kind.L_SQUARE, "Expected opening bracket")
        bounds: list[int] = []
        while not self._parse_optional_token(Token.Kind.R_SQUARE):
            bounds.append(self._parse_int_or_question())
            self._parse_optional_token(Token.Kind.COMMA)
        return bounds

    def _parse_tiled_stride(self) -> TiledStride:
        """
        tiled-stride ::= bounds `->` strides
        """
        bounds = self._parse_bound()
        self._parse_token(Token.Kind.ARROW, "Expected arrow")
        strides = self._parse_stride()
        if len(strides) != len(bounds):
            raise ParseError("Expected same number of strides and bounds")
        # construct the tiledstrides
        return TiledStride(
            [
                Stride(stride, bound)
                for stride, bound in zip(reversed(strides), reversed(bounds))
            ]
        )

    def parse(self) -> TiledStridedLayout:
        """
        tsl ::= tiled-stride (`,` tiled-stride)*` (, offset: ` offset)?
        """
        tstrides = []
        offset = 0
        while True:
            if self._current_token.kind == Token.Kind.GREATER:
                break
            if self.parse_optional_characters("offset"):
                self._parse_token(Token.Kind.COLON, "Expected colon")
                offset = self.parse_integer()
                break
            tstrides.append(self._parse_tiled_stride())
            self._parse_optional_token(Token.Kind.COMMA)
        return TiledStridedLayout(tstrides, offset=offset)
