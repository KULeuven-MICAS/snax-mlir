from abc import ABC
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic

from typing_extensions import Self, TypeVar, cast, deprecated, overload
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineConstantExpr,
    AffineDimExpr,
    AffineExpr,
    AffineMap,
)

from compiler.util.canonicalize_affine import canonicalize_map
from compiler.util.multiset import Multiset


@dataclass(frozen=True)
class AccessPattern(ABC):
    """
    Class specifying an access pattern for a single operand.
    This is specified by bounds and a pattern given as an AffineMap.
    """

    bounds: tuple[int | None, ...]
    pattern: AffineMap

    def __init__(self, bounds: Sequence[int | None], pattern: AffineMap):
        # Convert bounds to a tuple
        bounds = tuple(bounds)

        # Perform validations
        if len(bounds) != pattern.num_dims:
            raise ValueError(
                "The number of bounds should be equal to the dimension of the pattern"
            )

        if pattern.num_symbols > 0:
            raise ValueError("Symbols in the pattern are not supported")

        # Canonicalize the pattern
        new_pattern = canonicalize_map(pattern)

        # Assign attributes using object.__setattr__ due to frozen=True
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "pattern", new_pattern)

    @property
    def num_dims(self):
        return len(self.bounds)

    def disable_dims(self, dim: int) -> Self:
        """
        Returns an affine map with the leftmost `dim` dimensions set to 0

        For example:
            (d0, d1, d2) -> d0 + d1 + d2
        For `dim` = 1, will return:
            (d1, d2) -> d1 + d2
        For `dim` = 2, will return:
            (d2) -> d2
        """
        new_pattern = self.pattern.replace_dims_and_symbols(
            tuple(AffineConstantExpr(0) for _ in range(dim))
            + tuple(AffineDimExpr(i) for i in range(self.num_dims - dim)),
            [],
            self.num_dims - dim,
            0,
        )
        return type(self)(self.bounds[dim:], new_pattern)

    def depends_on(self, dim: int) -> bool:
        """
        Returns if this access pattern depends on a given dimension.

        For example:
            (d0, d1, d2) -> d0 + d1
            will return True for `dim` = 0 and 1, False for `dim` = 2
        """

        # helper function do determine if affine expression depends on something
        def expr_depends_on(expr: AffineExpr, dim: int):
            if isinstance(expr, AffineBinaryOpExpr):
                return expr_depends_on(expr.lhs, dim) or expr_depends_on(expr.rhs, dim)
            elif isinstance(expr, AffineDimExpr):
                return expr.position == dim
            return False

        return any(expr_depends_on(result, dim) for result in self.pattern.results)

    def __str__(self) -> str:
        return str(self.bounds) + str(self.pattern)


@dataclass(frozen=True)
class SchedulePattern(AccessPattern):
    """
    A schedule pattern is a pattern for a schedule of an operation.

    Schedule patterns are constrained to have static bounds.
    """

    # constrain bounds to only be int
    bounds: tuple[int, ...]

    def __init__(self, bounds: Sequence[int], pattern: AffineMap):
        if any(bound is None or bound <= 0 for bound in bounds):
            raise ValueError(
                "All bounds must be static, strictly positive integers for a schedule"
            )

        super().__init__(bounds, pattern)

    def rotate(self, dim: int) -> Self:
        """
        Returns a new schedule with the leftmost `dim` dimensions rotated

        For example:
            (d0, d1, d2) -> 1 * d0 + 2 * d1 + 3 * d2
        For `dim` = 3, will return:
            (d0, d1, d2) -> 3 * d0 + 1 * d1 + 2 * d2
        For `dim` = 2, will return:
            (d0, d1, d2) -> 2 * d0 + 1 * d1 + 3 * d2
        """

        # Rotate in the following manner:
        #     (0, 1, 2, 3, ..., dim-1, dim, dim+1, ..., num_dims - 1)
        # --> (1, 2, 3, ..., dim-1, 0, dim, dim+1, ..., num_dims - 1)

        new_dims = tuple(AffineDimExpr(i) for i in range(self.num_dims))
        new_dims = new_dims[dim - 1 : dim] + new_dims[: dim - 1] + new_dims[dim:]
        new_bounds = self.bounds[1:dim] + self.bounds[:1] + self.bounds[dim:]

        new_pattern = self.pattern.replace_dims_and_symbols(
            new_dims, [], self.num_dims, 0
        )
        return type(self)(new_bounds, new_pattern)

    def reorder(self, new_order: Sequence[int]) -> Self:
        """
        Returns a new SchedulePattern with dimensions reordered.

        For example:
            (d0, d1, d2) -> 1 * d0 + 2 * d1 + 3 * d2
        For `new_order` = (0, 2, 1), will return:
            (d0, d1, d2) -> 1 * d0 + 3 * d1 + 2 * d2
        For `new_order` = (1, 0, 2), will return:
            (d0, d1, d2) -> 2 * d0 + 1 * d1 + 3 * d2
        """
        new_dims = [AffineDimExpr(new_order[i]) for i in new_order]
        new_pattern = self.pattern.replace_dims_and_symbols(
            new_dims, [], self.num_dims, 0
        )
        new_bounds = [self.bounds[i] for i in new_order]
        return type(self)(new_bounds, new_pattern)

    def tile_dim(self, dim: int, template_bound: int) -> Self:
        """
        Returns a new access pattern with the `dim` dimension split up into two
        This translates to creating two for loops with adjusted bounds from one for loop


        For example:
            (d0, d1, d2) -> d0 + d1 + d2
        For `dim` = 1, `template_bound` = 2:
            (d0, d1, d2, d3) -> d0 + 2 * d1 + d2 + d3

        The bounds are split in similar fashion:
        For example:
            [2, 8, 2]
        For `dim` = 1, `template_bound` = 2:
            [2, 4, 2, 2]

        """
        transform_map = AffineMap(
            num_dims=self.num_dims + 1,
            num_symbols=0,
            # (d0, d1, d2, ..., dim-1) -> (d0, d1, d2, ..., dim-1)
            results=tuple(AffineDimExpr(i) for i in range(dim))
            # (dim) -> (template_bound * dim + dim + 1)
            + (AffineDimExpr(dim) * template_bound + AffineDimExpr(dim + 1),)
            # (dim + 1, dim + 2, ...) -> (dim + 2, dim + 3, dim + 3)
            + tuple(AffineDimExpr(i + 1) for i in range(dim + 1, self.num_dims)),
        )
        new_pattern = self.pattern.compose(transform_map)
        bound_to_tile = self.bounds[dim]
        tiled_bound = bound_to_tile // template_bound
        new_bounds = (
            self.bounds[:dim] + (tiled_bound, template_bound) + self.bounds[dim + 1 :]
        )

        return type(self)(new_bounds, new_pattern)

    def pad_dim(self, dim: int, template_bound: int) -> Self:
        """
        Returns a new schedule pattern with the `dim` dimension padded up
        to the template dimension
        """
        new_bounds = list(self.bounds)
        new_bounds[dim] = template_bound
        return type(self)(new_bounds, self.pattern)

    def add_dim(self) -> Self:
        """
        Returns a new schedule pattern with an extra empty dimension inserted.
        For example:
            (d0, d1) -> d0 + d1
        Will result in:
            (d0, d1, d2) -> d1 + d2
        """
        new_pattern = self.pattern
        transform_map = AffineMap(
            num_dims=self.num_dims + 1,
            num_symbols=0,
            results=tuple(AffineDimExpr(i + 1) for i in range(self.num_dims)),
        )
        new_pattern = self.pattern.compose(transform_map)
        new_bounds = (1,) + self.bounds
        return type(self)(new_bounds, new_pattern)


@dataclass(frozen=True)
class TemplatePattern(AccessPattern):
    """
    Template pattern is a pattern for an accelerator template.

    Templates should not be transformed through either tiling/rotating/others.

    Template bounds have some special properties:
        - bound == None -> bound can be programmed to anything from 0 to inf
        - bound <= 0: undefined behaviour
        - bound == 1: bound can be anything, but should be fixed to 1 (used for stationarity)
        - bound <= 1: bound is fixed
    """

    def __init__(self, bounds: Sequence[int | None], pattern: AffineMap):
        super().__init__(bounds, pattern)

    def matches(self, sp: SchedulePattern):
        """
        Check if a given schedule pattern matches this
        template pattern.
        """
        self_pattern = canonicalize_map(self.pattern, extreme=True)
        schedule_pattern = canonicalize_map(sp.pattern, extreme=True)
        return Multiset(self_pattern.results).is_subset(Multiset(schedule_pattern.results))


P = TypeVar("P", bound=AccessPattern)


class PatternCollection(Sequence[P], Generic[P], ABC):
    """
    Abstract base class for collections of AccessPatterns.
    Provides common methods and properties for Schedule and Template classes.
    """

    def __init__(self, patterns: Iterable[P]):
        self._patterns = tuple(patterns)

    @overload
    def __getitem__(self, index: int) -> P:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[P]:
        ...

    def __getitem__(self, index: int | slice):
        return self._patterns[index]

    def __len__(self) -> int:
        return len(self._patterns)

    def __iter__(self) -> Iterator[P]:
        return iter(self._patterns)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PatternCollection):
            return False
        return self._patterns == other._patterns

    @property
    @deprecated(
        "only valid in trivial cases (hindsight: no, this should always be valid)"
    )
    def num_dims(self) -> int:
        return self[0].num_dims

    @property
    def max_dim(self) -> int:
        return max(pattern.num_dims for pattern in self._patterns)

    @property
    def bounds(self) -> list[int | None]:
        """
        Returns the combined bounds of all patterns in this collection
        """
        result: list[int | None] = []
        for i in range(self.num_dims):
            combined_bounds = [pattern.bounds[i] for pattern in self._patterns]
            if any([x is None for x in combined_bounds]):
                result.append(None)
            else:
                combined_bounds = cast(list[int], combined_bounds)
                result.append(max(combined_bounds))
        return result

    def disable_dims(self, dim: int) -> Self:
        return type(self)(sp.disable_dims(dim) for sp in self)

    def clear_unused_dims(self, bounds: tuple[int] | None = None) -> Self:
        """
        Returns a PatternCollection of which all dimensions that have bound 1 are cleared.
        Optionally, specify custom bounds.
        """
        if bounds is None:
            pattern_bounds = tuple(self.bounds)
        else:
            pattern_bounds = bounds
        unused_dims = tuple(i for i, bound in enumerate(pattern_bounds) if bound == 1)
        dim_substitutions = []
        unused_counter = 0
        for dim in range(self.num_dims):
            if dim not in unused_dims:
                dim_substitutions.append(AffineDimExpr(dim - unused_counter))
            else:
                dim_substitutions.append(AffineConstantExpr(0))
                unused_counter += 1
        return type(self)(
            type(self._patterns[0])(
                tuple(
                    bound
                    for bound, pattern_bound in zip(sp.bounds, pattern_bounds)
                    if pattern_bound != 1
                ),
                sp.pattern.replace_dims_and_symbols(
                    dim_substitutions, [], self.num_dims - unused_counter, 0
                ),
            )
            for sp in self
        )

    def __str__(self) -> str:
        return "\n".join(str(pattern) for pattern in self._patterns)


class Schedule(PatternCollection[SchedulePattern]):
    """
    A schedule consisting of multiple SchedulePatterns for different operands.
    """

    def rotate(self, dim: int) -> Self:
        return type(self)(sp.rotate(dim) for sp in self)

    def reorder(self, new_order: Sequence[int]) -> Self:
        return type(self)(sp.reorder(new_order) for sp in self)

    def tile_dim(self, dim: int, template_bound: int) -> Self:
        return type(self)(sp.tile_dim(dim, template_bound) for sp in self)

    def pad_dim(self, dim: int, template_bound: int) -> Self:
        return type(self)(sp.pad_dim(dim, template_bound) for sp in self)

    def add_dim(self) -> Self:
        return type(self)(sp.add_dim() for sp in self)


class Template(PatternCollection[TemplatePattern]):
    """
    A template consisting of multiple TemplatePatterns for different operands.
    """

    def matches(self, schedule: Schedule):
        if len(schedule) != len(self):
            return False
        for sp, tp in zip(schedule, self):
            if not tp.matches(sp):
                return False
        return True
