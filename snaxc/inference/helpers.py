from collections.abc import Generator

from xdsl.dialects import func, llvm, scf
from xdsl.ir import Block, BlockArgument, Operation, OpResult, Region, SSAValue

from snaxc.dialects import accfg


def has_accfg_effects(op: Operation) -> bool:
    """
    Checks if an operation effects state managed by the accfg dialect, according to the following criteria:

    - If provided, an attribute named `accfg.effects` of either of the two types will overwrite inference.
      These attributes will need to be added by the provider of the IR, as only they know which functions may affect
      the accelerator.
    - By default we assume that function calls modify state (e.g. `func.call` and `llvm.call`).
    - All other operation don't modify state, as long as the ops contained within them don't modify state according
      to above criterion.
    """
    # check if op is marked as effecting
    effects_attr = op.attributes.get("accfg.effects", None)
    if isinstance(effects_attr, accfg.EffectsAttr):
        return effects_attr.effects != accfg.EffectsEnum.NONE

    # ops that may affect state are function calls
    # all function calls that *don't* effect must be marked
    if isinstance(op, func.CallOp | llvm.CallOp):
        return True

    # Recurse into children to check them according to the same rules
    if any(
        has_accfg_effects(op)
        for region in op.regions
        for block in region.blocks
        for op in block.ops
    ):
        return True
    # all other ops are assumed to not have effects
    return False


def get_initial_value_for_scf_for_lcv(loop: scf.ForOp, var: SSAValue) -> SSAValue:
    """
    Given a loop-carried variable inside an scf for loop as the block argument,
    return the SSA value that is passed as the initial value to it.
    """
    if var not in loop.body.block.args:
        raise ValueError(
            f"Given value {var} not a block argument of the for loop {loop}!"
        )
    idx = loop.body.block.args.index(var) - 1
    return loop.iter_args[idx]


def val_is_defined_in_block(val: SSAValue, block: Block) -> bool:
    """
    Check if val is defined in block or in blocks nested in block.
    """
    if isinstance(val, BlockArgument):
        block_ptr: Block | None = val.owner
        # walk upwards until we hit block or None
        while block_ptr is not None and block_ptr != block:
            # walk upwards
            block_ptr = block_ptr.parent_block()
        # we either ran out of blocks
        return block_ptr is not None
    elif isinstance(val, OpResult):
        op_ptr: Operation | None = val.owner
        # walk up until we either hit the right block, or run out of parents
        while op_ptr is not None and op_ptr.parent_block() != block:
            op_ptr = op_ptr.parent_op()
        # iff we didn't hit the right block, ptr must be none
        return op_ptr is not None
    else:
        raise ValueError(f"Unsupported SSA Value type: {val}")


def calc_if_state_delta(
    old_state: dict[str, SSAValue],
    if_state: dict[str, SSAValue],
    else_state: dict[str, SSAValue],
) -> dict[str, tuple[SSAValue, SSAValue]]:
    """
    Given three state dictionaries (mapping accelerator names
    to SSA vals containing their state, return a new dict that:

    - Contains tuples (if_branch_val, else_branch_val)
    - For all accelerators whose state val changed in *at least
      one* of the branches
    - And for all accelerator states that got introduced in *both*
      branches
    """
    new_state: dict[str, tuple[SSAValue, SSAValue]] = {}

    # for every key in the old state, find out if it changed
    for k in old_state:
        # get new vals (or None if dropped)
        new_vals = (
            if_state.pop(k, None),
            else_state.pop(k, None),
        )
        # drop val if it is invalidated on one side
        if any(v is None for v in new_vals):
            continue
        # if no val changed
        if all(v == old_state[k] for v in new_vals):
            continue
        # pyright can't infer that we actually checked the argument type:
        new_state[k] = new_vals  # pyright: ignore[reportArgumentType]

    # check for states that are present in both branches
    for k in if_state:
        if k not in else_state:
            continue
        # add them to the new dict
        new_state[k] = (if_state[k], else_state[k])

    return new_state


def find_all_acc_names_in_region(reg: Region) -> set[str]:
    """
    Walk a region and return a set of accelerator names are set up in that region.
    """
    acs: set[str] = set()
    for op in reg.walk():
        if isinstance(op, accfg.SetupOp):
            acs.add(op.accelerator.data)
    return acs


def find_existing_block_arg(block: Block, accel: str) -> BlockArgument | None:
    """
    Inspect a block for block arguments of the correct accfg.StateType type, return the block arg if found.
    """
    for arg in block.args:
        if isinstance(arg.type, accfg.StateType) and arg.type.accelerator.data == accel:
            return arg
    return None


def previous_ops_of(op: Operation) -> Generator[Operation, None, None]:
    """
    Walks up the block and yields the ops preceeding `op` until the start of the block.

    Does not recurse into operations.
    """
    while op.prev_op is not None:
        yield op.prev_op
        op = op.prev_op


def iter_ops_range(
    start_op: Operation | None, end_op: Operation | None
) -> Generator[Operation, None, None]:
    """
    Create an iterator that goes from start_op (inclusive) to end_op (exclusive).

    If start_op is None, iterate from the start of the block.

    If end_op is None, iterate to the end of the block (inclusive).

    Both ops cannot be none.

    If provided, both start and end op must be in the same block.
    """

    if start_op is None:
        assert end_op is not None, "Can't have both start and end_op be None!"
        parent = end_op.parent_block()
        assert parent is not None, "Provided operations must have parent blocks!"
        start_op = parent.first_op

    assert start_op is not None  # just for pyright

    if end_op is not None:
        assert (
            start_op.parent_block() is end_op.parent_block()
        ), "Cannot iterate between operations not within the same block!"

    yield start_op
    while start_op.next_op is not end_op:
        assert start_op.next_op is not None  # just for pyright
        yield start_op.next_op
        start_op = start_op.next_op
