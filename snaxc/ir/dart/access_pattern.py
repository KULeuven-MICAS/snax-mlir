from abc import ABC
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, cast

from typing_extensions import Self, TypeVar, overload
from xdsl.ir.affine import AffineDimExpr, AffineMap

from snaxc.ir.dart.affine_transform import AffineTransform


@dataclass(frozen=True)
class AccessPattern(ABC):
    """
    Class specifying an access pattern for a single operand.
    This is specified by bounds and a pattern given as an AffineMap.
    """

    bounds: tuple[int | None, ...]
    pattern: AffineTransform

    def __init__(
        self, bounds: Sequence[int | None], pattern: AffineMap | AffineTransform
    ):
        # Convert bounds to a tuple
        bounds = tuple(bounds)

        if isinstance(pattern, AffineMap):
            pattern = AffineTransform.from_affine_map(pattern)

        # Perform validations
        if len(bounds) != pattern.num_dims:
            raise ValueError(
                "The number of bounds should be equal to the dimension of the pattern"
            )

        # Assign attributes using object.__setattr__ due to frozen=True
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "pattern", pattern)

    @property
    def num_dims(self):
        return len(self.bounds)

    def inner_dims(self, dim: int) -> Self:
        """
        Returns an affine map with all but the innermost `dim` dimensions set to 0

        For example:
            (d0, d1, d2) -> d0 + d1 + d2
        For `dim` = 2, will return:
            (d1, d2) -> d1 + d2
        For `dim` = 1, will return:
            (d2) -> d2
        """
        if dim <= 0:
            raise ValueError("can only select a positive number of dimensions")
        return type(self)(
            self.bounds[-dim:],
            AffineTransform(self.pattern.A[:, -dim:], self.pattern.b),
        )

    def __str__(self) -> str:
        bounds = [str(b) if b else "?" for b in self.bounds]
        return f'({", ".join(bounds)}) {str(self.pattern.to_affine_map())}'


@dataclass(frozen=True)
class SchedulePattern(AccessPattern):
    """
    A schedule pattern is a pattern for a schedule of an operation.

    Schedule patterns are constrained to have static bounds.
    """

    # constrain bounds to only be int
    bounds: tuple[int, ...]

    def __init__(self, bounds: Sequence[int], pattern: AffineMap | AffineTransform):
        if any(bound <= 0 for bound in bounds):
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
        return AccessPattern()
        """

        # Rotate in the following manner:
        #     (0, 1, 2, 3, ..., dim-1, dim, dim+1, ..., num_dims - 1)
        # --> (1, 2, 3, ..., dim-1, 0, dim, dim+1, ..., num_dims - 1)

        new_bounds = self.bounds[1:dim] + self.bounds[:1] + self.bounds[dim:]
        new_a = self.pattern.A[:, [*range(1, dim), 0, *range(dim, self.num_dims)]]
        new_pattern = AffineTransform(new_a, self.pattern.b)

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
        transform_map = AffineTransform.from_affine_map(
            AffineMap(
                num_dims=self.num_dims + 1,
                num_symbols=0,
                # (d0, d1, d2, ..., dim-1) -> (d0, d1, d2, ..., dim-1)
                results=tuple(AffineDimExpr(i) for i in range(dim))
                # (dim) -> (template_bound * dim + dim + 1)
                + (AffineDimExpr(dim) * template_bound + AffineDimExpr(dim + 1),)
                # (dim + 1, dim + 2, ...) -> (dim + 2, dim + 3, dim + 3)
                + tuple(AffineDimExpr(i + 1) for i in range(dim + 1, self.num_dims)),
            )
        )
        new_pattern = self.pattern.compose(transform_map)
        bound_to_tile = self.bounds[dim]
        tiled_bound = bound_to_tile // template_bound
        new_bounds = (
            self.bounds[:dim] + (tiled_bound, template_bound) + self.bounds[dim + 1 :]
        )

        return type(self)(new_bounds, new_pattern)

    def add_dim(self) -> Self:
        """
        Returns a new schedule pattern with an extra empty dimension inserted.
        For example:
            (d0, d1) -> d0 + d1
        Will result in:
            (d0, d1, d2) -> d1 + d2
        """
        new_pattern = self.pattern
        transform_map = AffineTransform.from_affine_map(
            AffineMap(
                num_dims=self.num_dims + 1,
                num_symbols=0,
                results=tuple(AffineDimExpr(i + 1) for i in range(self.num_dims)),
            )
        )
        new_pattern = self.pattern.compose(transform_map)
        new_bounds = (1,) + self.bounds
        return type(self)(new_bounds, new_pattern)


@dataclass(frozen=True)
class TemplatePattern(AccessPattern):
    """
    Template pattern is a pattern for an accelerator template.

    Templates should not be transformed through either tiling/rotating/others.
    """

    def __init__(
        self, bounds: Sequence[int | None], pattern: AffineMap | AffineTransform
    ):
        super().__init__(bounds, pattern)

    def matches(self, sp: SchedulePattern):
        """
        Check if a given schedule pattern matches this
        template pattern.
        """
        if sp.num_dims > self.num_dims:
            sp = sp.inner_dims(self.num_dims)
        elif sp.num_dims < self.num_dims:
            return False
        if sp.pattern != self.pattern:
            return False
        return True


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
        other = cast(PatternCollection[P], other)
        return self._patterns == other._patterns

    @property
    def num_dims(self) -> int:
        return self[0].num_dims

    @property
    def max_dim(self) -> int:
        return max(pattern.num_dims for pattern in self._patterns)

    def inner_dims(self, dim: int) -> Self:
        return type(self)(sp.inner_dims(dim) for sp in self)

    def clear_unused_dims(self, bounds: tuple[int] | None = None) -> Self:
        """
        Returns a PatternCollection of which all dimensions that have bound 1 are cleared.
        Optionally, specify custom bounds.
        """
        if bounds is None:
            pattern_bounds = self._patterns[0].bounds
        else:
            pattern_bounds = bounds
        used_dims = tuple(i for i, bound in enumerate(pattern_bounds) if bound != 1)
        return type(self)(
            type(self._patterns[0])(
                tuple(bound for bound in pattern_bounds if bound != 1),
                AffineTransform(sp.pattern.A[:, used_dims], sp.pattern.b),
            )
            for sp in self
        )

    def __str__(self) -> str:
        return "\n".join(str(pattern) for pattern in self)


class Schedule(PatternCollection[SchedulePattern]):
    """
    A schedule consisting of multiple SchedulePatterns for different operands.
    """

    def rotate(self, dim: int) -> Self:
        return type(self)(sp.rotate(dim) for sp in self)

    def tile_dim(self, dim: int, template_bound: int) -> Self:
        return type(self)(sp.tile_dim(dim, template_bound) for sp in self)

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
