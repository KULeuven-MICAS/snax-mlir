from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass

from xdsl.dialects import scf
from xdsl.ir import Block, Operation, SSAValue, Use
from xdsl.rewriter import InsertPoint
from xdsl.traits import IsTerminator


@dataclass(frozen=True, slots=True)
class InsertCandidate:
    idx: int
    op: Operation


def uses_through_controlflow(val: SSAValue) -> Generator[Use, None, None]:
    """
    Find all uses of an SSA value, even through control flow. For example, given the IR:

    ```
    %a = arith.constant 1 : i32

    %res = scf.for (...) iter_args(%x = %a : i32) {
        %loop = arith.add ..., %x
        yield %x
    }

    arith.add %a, ...
    ```

    uses_through_controlflow(%a) will yield:
        - scf.for
        - %loop = arith.add ..., %x
        - yield %x
        - arith.add %a, ...
    """
    for use in val.uses:
        # return each use
        yield use
        # if it's an scf.for, recurse on block arg
        if isinstance(use.operation, scf.For):
            for_op = use.operation
            yield from uses_through_controlflow(for_op.body.block.args[use.index - 2])
        # if it's a yield, recurse on parent op result
        elif isinstance(use.operation, scf.Yield):
            yield from uses_through_controlflow(
                use.operation.parent_op().results[use.index]
            )


def get_insertion_points_where_val_dangles(
    vals: Sequence[SSAValue],
) -> Iterable[InsertPoint]:
    """
    Return all insertion points for positions where a value starts dangling.

    If val is used by a terminator, it's not assumed dangling in that block.

    If multiple vals are given, return the earliest point in each block where
    all vals used in that block are dangling.
    """
    # first filter out vals that don't have any uses
    remaining_vals = list(vals)
    for val in tuple(vals):
        # if no uses are found, just return the first position where the value is live:
        if not val.uses:
            if isinstance(val.owner, Operation):
                yield InsertPoint.after(val.owner)
            else:
                yield InsertPoint.at_start(val.owner)
            remaining_vals.remove(val)

    # remember insertion candidates and their positions
    # we need to keep track of the last use of the value in each block it's used in.
    inserts: dict[Block, InsertCandidate] = dict()
    dead_blocks: set[Block] = set()
    # check each use
    for val in remaining_vals:
        for use in val.uses:
            # grab the block
            block = use.operation.parent_block()
            if block is None or block in dead_blocks:
                continue

            # grab the candidate for this block
            candidate = inserts.get(block, None)
            # and the position of the current op in the block
            idx = block.get_operation_index(use.operation)

            # if there is a candidate
            if candidate is not None:
                # and the current op comes before the candidate
                if idx < candidate.idx:
                    continue  # skip it
            # if we are a terminator
            if use.operation.has_trait(IsTerminator):
                # don't insert in this block
                inserts.pop(block, None)
                # never insert in this block
                dead_blocks.add(block)
                continue

            # put insertion candidate into candidate dict
            inserts[block] = InsertCandidate(idx, use.operation)

    # return all insertion candidates that we have
    yield from (InsertPoint.after(cd.op) for cd in inserts.values())
