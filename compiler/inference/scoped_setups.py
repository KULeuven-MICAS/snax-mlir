"""
This file contains helpers that allow us to reason about "scoped setups"

A scoped setup is a setup and all associated operations needed to calculate its values. It can be
thought of as the sequence of operations needed to set up the accelerator. Given the following IR:

```
func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
  %0 = arith.constant 0 : index
  %cst_0 = arith.constant 0 : i5
  %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
  %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
  %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
  %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index

  %5 = accfg.setup "snax_hwpe_mult" to ("A" = %1 : index, "B" = %2 : index, "O" = %3 : index, "size" = %4 : index) : ...

  // ...
```

The scoped setup of %5 would be:
```
ScopedSetupWithInputs(
    setup = %5 = accfg.setup ...
    dependent_vars = tuple(%arg0, %arg1, %arg2)
    inputs = (
        %0 = arith.constant 0 : index
        %1 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
        %2 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
        %3 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
        %4 = "memref.dim"(%arg0, %0) : (memref<?xi32>, index) -> index
    )
)
```

With `%cst_0` being absent, as it's not used as part of the setup sequence.

Some further restrictions:

- A scoped setup is always scoped to within a block, no operations outside that block are considered.
- A scoped setup may only contain operations that are side effect free, as we only care to argue about
  setup sequences that we can move around or copy.

A scoped setup can be:

- Copied with new inputs
- Moved
- Erased
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from xdsl.ir import Block, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint
from xdsl.traits import is_side_effect_free

from compiler.dialects import accfg
from compiler.inference.helpers import val_is_defined_in_block


def get_scoped_setup_inputs(
    setup_op: accfg.SetupOp, scope: Block
) -> ScopedSetupWithInputs | None:
    """
    Takes a setup op and a set of known input variables, and looks at all inputs to the setup op to determine
    """
    input_vars = scope.args
    inputs = []

    # value inspection starts at all inputs to the setup op:
    vals_to_inspect = [*setup_op.values]
    # until we are out of values to inspect, we:
    while vals_to_inspect:
        # grab a value to inspect
        val = vals_to_inspect.pop(0)
        # check if it's one of the loop dependent variables we are looking at
        if val in input_vars:
            # if it is, we have walked the use-def chain to its conclusion
            continue
        # if the value is defined outside our scope, we don't need to dig any further
        if not val_is_defined_in_block(val, scope):
            continue
        # if it's another block argument, we just give up
        if isinstance(val.owner, Block):
            # shrug, idk what to do here
            print(
                f"constructung pure input tree for setup {setup_op} failed on block argument {val} of:\n"
                f"{val.owner.parent_op()}",
                file=sys.stderr,
            )
            return None
        # if it's an operation
        elif isinstance(val.owner, Operation):
            # we check that it's effect free
            if is_side_effect_free(val.owner):
                # if it is effect free, we recurse on it's operands
                vals_to_inspect.extend(val.owner.operands)
                # and note the operation down as one that computes our input variables
                inputs.append(val.owner)
            else:
                # impure operations represent a problem for us. Print a warning.
                print(
                    f"Op with effects in use-def chain upwards of setup {setup_op}: {val.owner}",
                    file=sys.stderr,
                )

    return ScopedSetupWithInputs(
        setup_op, input_vars, tuple(reversed(inputs))  # reverse order
    )


@dataclass
class ScopedSetupWithInputs:
    """
    This dataclass represents a setup inside a loop that is loop variable dependent.

    - setup: The setup operation
    - dependent_vars: SSA values that we know are loop-dependent
    - inputs: A sequence of Operations that are dependent on the depdendent_var and need to be moved with the setup op

    Has three helpers to:

    1. Insert all the input ops and setup op at a specific position
    2. Clone everything with new input ops
    3. Erase the setup op
    4. Move the setup and inputs to be at least above a certain point
    """

    setup: accfg.SetupOp
    dependent_vars: tuple[SSAValue, ...]
    inputs: tuple[Operation, ...]

    def insert_at_position(self, rewriter: PatternRewriter, point: InsertPoint):
        rewriter.insert_op((*self.inputs, self.setup), point)

    def copy_with_new_dependent_vals(
        self: ScopedSetupWithInputs, new_dependent_vars: tuple[SSAValue, ...]
    ) -> ScopedSetupWithInputs:
        """
        Copy the scoped setup and replace all inputs with new inputs, replacing dependent_vars with new_dependent_vars.

        Returns a new scoped setup.
        """
        ops = []
        # map the old inputs to the new inputs
        mapper: dict[SSAValue, SSAValue] = dict(
            zip(self.dependent_vars, new_dependent_vars, strict=True)
        )
        # for each operation that calculates the setup input:
        for op in self.inputs:
            # clone it, replacing the old inputs with the new inputs
            new_op = op.clone(value_mapper=mapper)
            # use the results of the new op in place of the old ops results
            for old, new in zip(op.results, new_op.results):
                mapper[old] = new
            ops.append(new_op)

        return ScopedSetupWithInputs(
            self.setup.clone(value_mapper=mapper), new_dependent_vars, tuple(ops)
        )

    def erase(self, replacement_state: SSAValue, rewriter: PatternRewriter):
        """
        Erase the setup op, keeping the inputs in place (they can be erased by cse)
        """
        self.setup.out_state.replace_by(replacement_state)
        rewriter.erase_op(self.setup)

    def lazy_move_up(self, scope: Block, pt: InsertPoint, rewriter: PatternRewriter):
        """
        This method will move operations such that all inputs are located above the insertion point.

        Operations that are already above the insertion point won't be moved.
        """
        assert (
            pt.insert_before is not None
        ), "can't move to end of block! (malformed IR)"
        assert (
            pt.insert_before.parent_block() is scope
        ), "Can't move operations to an insertion point that is not directly in $scope!"

        # don't do anything if the insertion point is one of our ops:
        if pt.insert_before in self.inputs or pt.insert_before is self.setup:
            return

        positions: dict[Operation, int] = dict(
            (op, i) for i, op in enumerate(scope.ops)
        )

        insertion_point_position = positions[pt.insert_before]

        # only move operations that are below the insertion point
        for op in (*self.inputs, self.setup):
            assert op in positions
            idx = positions[op]
            if idx <= insertion_point_position:
                continue
            op.detach()
            rewriter.insert_op(op, pt)
