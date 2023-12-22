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

    def _parse_step(self) -> list[int]:
        """
        steps ::== `(` steps (`,` steps)* `)`
        """
        self._parse_token(Token.Kind.L_PAREN, "Expected opening bracket")
        steps: list[int] = []
        while not self._parse_optional_token(Token.Kind.R_PAREN):
            steps.append(self._parse_int_or_question())
            self._parse_optional_token(Token.Kind.COMMA)
        return steps

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
        bounds = self._parse_bound()
        self._parse_token(Token.Kind.ARROW, "Expected arrow")
        steps = self._parse_step()
        if len(steps) != len(bounds):
            raise ParseError("Expected same number of steps and bounds")
        # construct the tiledstrides
        return TiledStride([Stride(step, bound) for step, bound in zip(steps, bounds)])

    def parse(self) -> TiledStridedLayout:
        """
        tsl ::= `(` tiled-stride (`,` tiled-stride)* `)`
        """
        self._parse_token(Token.Kind.L_PAREN, "Expected opening bracket")
        tstrides = []
        while not self._parse_optional_token(Token.Kind.R_PAREN):
            tstrides.append(self._parse_tiled_stride())
            self._parse_optional_token(Token.Kind.COMMA)
        return TiledStridedLayout(tstrides)
