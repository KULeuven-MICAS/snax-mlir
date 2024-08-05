from collections.abc import Sequence

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)


class LaTeXPrinter:
    result: str = ""

    # create new printing functions:
    @staticmethod
    def print_dim(expr: AffineDimExpr, dims: Sequence[str] | None = None):
        if not dims:
            return f"d{expr.position}"
        return dims[expr.position]

    @staticmethod
    def print_bin(expr: AffineBinaryOpExpr, dims: Sequence[str] | None = None):
        if expr.kind == AffineBinaryOpKind.Mul:
            # print constant first
            if isinstance(expr.rhs, AffineConstantExpr):
                return rf"{LaTeXPrinter.print_expr(expr.rhs, dims)} \cdot {LaTeXPrinter.print_expr(expr.lhs, dims)}"
            return rf"{LaTeXPrinter.print_expr(expr.lhs, dims)} \cdot {LaTeXPrinter.print_expr(expr.rhs, dims)}"
        return (
            f"{LaTeXPrinter.print_expr(expr.lhs, dims)} "
            + f"{expr.kind.get_token()} "
            + f"{LaTeXPrinter.print_expr(expr.rhs, dims)}"
        )

    @staticmethod
    def print_expr(expr: AffineExpr, dims: Sequence[str] | None = None):
        if isinstance(expr, AffineBinaryOpExpr):
            return LaTeXPrinter.print_bin(expr, dims)
        if isinstance(expr, AffineDimExpr):
            return LaTeXPrinter.print_dim(expr, dims)
        return str(expr)

    @staticmethod
    def print_map(
        map: AffineMap,
        dims: Sequence[str] | None = None,
        bounds: Sequence[int | None] | None = None,
        phantomize: bool = False,
    ):
        if not dims:
            dims = ["d" + str(i) for i in range(map.num_dims)]
        results = r", \,".join(LaTeXPrinter.print_expr(expr, dims) for expr in map.results)
        if bounds:
            dims = [rf"\overset{{{bound}}}{{\bar{{{dim}}}}}" if bound else str(dim) for dim, bound in zip(dims, bounds)]
        dims_result = r", \,".join(dims)

        if phantomize:
            return rf"$\phantom{{({dims_result}) \rightarrow}} ({results})$"
        else:
            return rf"$({dims_result}) \rightarrow ({results})$"

    @staticmethod
    def print_maps(
        maps: Sequence[AffineMap],
        dims: Sequence[str] | None = None,
        bounds: Sequence[int | None] | None = None,
    ):
        return "\n\n".join(LaTeXPrinter.print_map(map, dims, bounds, i > 0) for i, map in enumerate(maps))

    start_file = r"""

\documentclass{article}

\usepackage{amsmath}

\pagenumbering{gobble}

\begin{document}

    """

    end_file = r"""
\end{document}
    """

    def print(
        self,
        value: Sequence[AffineMap] | AffineMap,
        dims: Sequence[str] | None = None,
        bounds: Sequence[int | None] | None = None,
        comment: str | None = None,
    ):
        self.result += "\n\n"
        if comment:
            self.result += "%" + comment + "\n"
        if isinstance(value, Sequence):
            self.result += LaTeXPrinter.print_maps(value, dims, bounds)
        else:
            self.result += LaTeXPrinter.print_map(value, dims, bounds)
        self.result += "\n\n" + r"\newpage" + "\n\n"

    def get_result(self) -> str:
        return LaTeXPrinter.start_file + self.result + LaTeXPrinter.end_file
