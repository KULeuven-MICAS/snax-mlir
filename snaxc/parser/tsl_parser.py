from __future__ import annotations

from xdsl.parser import BaseParser, ParserState
from xdsl.utils.exceptions import ParseError
from xdsl.utils.mlir_lexer import MLIRTokenKind

from snaxc.ir.tsl import Stride, TiledStride, TiledStridedLayout


class TSLParser(BaseParser):
    def __init__(self, state: ParserState[MLIRTokenKind]) -> None:
        self._resume_from(state)

    def _parse_int_or_question(self, context_msg: str = "") -> int | None:
        """Parse either an integer literal, or a '?'."""
        if self._parse_optional_token(MLIRTokenKind.QUESTION) is not None:
            return None
        if (v := self.parse_optional_integer(allow_boolean=False)) is not None:
            return v
        self.raise_error("Expected an integer literal or `?`" + context_msg)

    def _parse_step(self) -> list[int | None]:
        """
        steps ::== `(` steps (`,` steps)* `)`
        """
        self._parse_token(MLIRTokenKind.L_PAREN, "Expected opening bracket")
        steps: list[int | None] = []
        while not self._parse_optional_token(MLIRTokenKind.R_PAREN):
            steps.append(self._parse_int_or_question())
            self._parse_optional_token(MLIRTokenKind.COMMA)
        return steps

    def _parse_bound(self) -> list[int | None]:
        """
        bounds ::== `[` bound (`,` bound)* `]`
        """
        self._parse_token(MLIRTokenKind.L_SQUARE, "Expected opening bracket")
        bounds: list[int | None] = []
        while not self._parse_optional_token(MLIRTokenKind.R_SQUARE):
            bounds.append(self._parse_int_or_question())
            self._parse_optional_token(MLIRTokenKind.COMMA)
        return bounds

    def _parse_tiled_stride(self) -> TiledStride:
        """
        tiled-stride ::= bounds `->` strides
        """
        bounds = self._parse_bound()
        self._parse_token(MLIRTokenKind.ARROW, "Expected arrow")
        steps = self._parse_step()
        if len(steps) != len(bounds):
            raise ParseError(self._current_token.span, "Expected same number of steps and bounds")
        # construct the tiledstrides
        return TiledStride([Stride(step, bound) for step, bound in zip(steps, bounds)])

    def parse(self) -> TiledStridedLayout:
        """
        tsl ::= tiled-stride (`,` tiled-stride)*` (, offset: ` offset)?
        """
        tstrides: list[TiledStride] = []
        offset = 0
        while True:
            if self._current_token.kind == MLIRTokenKind.GREATER:
                break
            if self.parse_optional_characters("offset"):
                self._parse_token(MLIRTokenKind.COLON, "Expected colon")
                offset = self.parse_integer()
                break
            tstrides.append(self._parse_tiled_stride())
            self._parse_optional_token(MLIRTokenKind.COMMA)
        return TiledStridedLayout(tstrides, offset=offset)
