from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

from typing_extensions import Self, deprecated
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from compiler.util.canonicalize_affine import canonicalize_map


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

    def rotate(self, dim: int) -> Self:
        """
        Returns a new access pattern with the leftmost `dim` dimensions rotated

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

        new_dims = tuple(AffineDimExpr(i) for i in range(self.num_dims))
        new_dims = new_dims[1:dim] + new_dims[:1] + new_dims[dim:]
        new_bounds = self.bounds[1:dim] + self.bounds[:1] + self.bounds[dim:]

        new_pattern = self.pattern.replace_dims_and_symbols(
            new_dims, [], self.num_dims, 0
        )
        return type(self)(new_bounds, new_pattern)

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
        tiled_bound = bound_to_tile // template_bound if bound_to_tile else None
        new_bounds = (
            self.bounds[:dim] + (tiled_bound, template_bound) + self.bounds[dim + 1 :]
        )

        return type(self)(new_bounds, new_pattern)


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


@dataclass(frozen=True)
class TemplatePattern(AccessPattern):
    """
    Template pattern is a pattern for an accelerator template.

    Templates should not be transformed through either tiling/rotating/others.
    """

    def __init__(self, bounds: Sequence[int], pattern: AffineMap):
        super().__init__(bounds, pattern)

    def tile_dim(self, dim: int, template_bound: int) -> Self:
        raise RuntimeError("A template should not be tiled")

    def rotate(self, dim: int) -> Self:
        raise RuntimeError("A template should not be rotated")

    def matches(self, sp: SchedulePattern):
        """
        Check if a given schedule pattern matches this
        template pattern.
        """
        if sp.num_dims != self.num_dims:
            return False
        if sp.pattern != self.pattern:
            return False
        return True


class Schedule(tuple[SchedulePattern]):
    @property
    @deprecated("only valid in trivial cases")
    def num_dims(self):
        return self[0].num_dims

    def rotate(self, dim: int) -> Self:
        return type(self)(sp.rotate(dim) for sp in self)

    def disable_dims(self, dim: int) -> Self:
        return type(self)(sp.disable_dims(dim) for sp in self)

    def tile_dim(self, dim: int, template_bound: int) -> Self:
        return type(self)(sp.tile_dim(dim, template_bound) for sp in self)


class Template(tuple[TemplatePattern]):
    @property
    @deprecated("only valid in trivial cases")
    def num_dims(self):
        return self[0].num_dims

    def disable_dims(self, dim: int) -> Self:
        return type(self)(tp.disable_dims(dim) for tp in self)

    def matches(self, schedule: Schedule):
        if len(schedule) != len(self):
            return
        for sp, tp in zip(schedule, self):
            if not tp.matches(sp):
                return False
        return True
