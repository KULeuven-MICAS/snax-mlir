from abc import ABC
from collections.abc import Iterable, Sequence

from xdsl.dialects import arith, llvm
from xdsl.dialects.builtin import IntegerAttr, i64
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.accelerator import Accelerator
from compiler.dialects import accfg
from compiler.inference.trace_acc_state import infer_state_of


class RoCCAccelerator(Accelerator, ABC):
    """
    Abstract base class for extending AcceleratorInterfaces
    with common RoCC lowerings.
    """

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        """
        There are no accelerator barrier operations for RoCC
        """
        return []

    @staticmethod
    def lower_acc_launch(
        launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
        xcustom_acc = 3  # hardcoded to 3 for now
        vals = create_pairs(launch_op)
        # Create the sequence of all operations that need to be emitted
        return combine_pairs_to_ops(acc_op.launch_field_items(), vals, xcustom_acc)

    @staticmethod
    def lower_acc_setup(
        setup_op: accfg.SetupOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
        # If you have the first op, materialize a default value for the register values
        # which are not set yet, otherwise create_pairs might not retrace a previous
        # value
        # Note: This assumes that the full pair was previously set, but it was
        #   deduplicated. if one of the two values of the pair was not set, this fix
        #   can not recover the previous value
        optional_default_value = []
        to_add_as_defaults: list[str] = []
        if setup_op.in_state is None:
            field_dict = dict(setup_op.iter_params())
            instructions = set([name[:-4] for name, _ in setup_op.iter_params()])
            for instruction in instructions:
                if instruction + ".rs1" not in field_dict:
                    to_add_as_defaults.append(instruction + ".rs1")
                if instruction + ".rs2" not in field_dict:
                    to_add_as_defaults.append(instruction + ".rs2")
            # If there are additional defaults to be added, replace the current setup
            # op with a new one that uses the defaults
            if to_add_as_defaults:
                optional_default_value.append(
                    default_val := arith.ConstantOp.from_int_and_width(0, i64)
                )
                new_params = list(field_dict.keys()) + to_add_as_defaults
                new_values = list(field_dict.values()) + [default_val] * len(
                    to_add_as_defaults
                )
                setup_op = accfg.SetupOp(new_values, new_params, setup_op.accelerator)

        xcustom_acc = 3  # hardcoded to 3 for now
        vals = create_pairs(setup_op)
        # Only pass on the field names that are set in the current setup
        instructions = set([name[:-4] for name, _ in setup_op.iter_params()])
        current_fields = {
            key: val for key, val in acc_op.field_items() if key[:-4] in instructions
        }.items()
        # Create the sequence of all operations that need to be emitted
        return [
            *optional_default_value,
            *combine_pairs_to_ops(current_fields, vals, xcustom_acc),
        ]


def create_pairs(
    fields_op: accfg.LaunchOp | accfg.SetupOp,
) -> dict[str, tuple[SSAValue, SSAValue]]:
    """
    For a given RoCC launch or setup op, return a map that maps a single RoCC
    instruction to pairs of two SSAValues

    For setup ops, this can retrace back previous setup state if necessary
    (i.e. if one of the two operands of an instruction gets dedupped)
    """
    # Make a set of all the unique instruction names in the current operation
    instructions = set([name[:-4] for name, _ in fields_op.iter_params()])
    field_dict = dict(fields_op.iter_params())

    # For setup_ops, get the previous setup state, if necessary
    if isinstance(fields_op, accfg.SetupOp):
        prev_state = infer_state_of(fields_op.in_state) if fields_op.in_state else {}
        for instruction in instructions:
            if instruction + ".rs1" not in field_dict:
                field_dict[instruction + ".rs1"] = prev_state[instruction + ".rs1"]
            if instruction + ".rs2" not in field_dict:
                field_dict[instruction + ".rs2"] = prev_state[instruction + ".rs2"]
    # For launch_ops, no tracing back is necessary, since dedup doesn't happen
    elif isinstance(fields_op, accfg.LaunchOp):
        pass
    # Assert that pairs of rs1 and rs2 exist for each item in the fields
    for instruction in instructions:
        assert instruction + ".rs1" in field_dict, f"No rs1 found for {instruction}"
        assert instruction + ".rs2" in field_dict, f"No rs2 found for {instruction}"
    # Create a dictionary that contains the two vals associated
    # to each single RoCC instruction
    instruction_map: dict[str, tuple[SSAValue, SSAValue]] = {}
    for instruction in instructions:
        instruction_map.setdefault(
            instruction,
            (field_dict[instruction + ".rs1"], field_dict[instruction + ".rs2"]),
        )
    return instruction_map


def combine_pairs_to_ops(
    field_items: Iterable[tuple[str, IntegerAttr]],
    values: dict[str, tuple[SSAValue, SSAValue]],
    xcustom_acc: int,
):
    ops: Sequence[Operation] = []
    """
    Emits a custom RoCC instruction for each field_item in field_items
    CUSTOM field can be specified by xcustom_acc.
    """
    for name, func7 in [
        (name[:-4], func7.value.data)
        for name, func7 in field_items
        if name.endswith(".rs1")
    ]:
        ops.extend(
            [
                get_rocc_inline_asm(
                    str(xcustom_acc),
                    str(func7),
                    values[name][0],
                    values[name][1],
                ),
            ]
        )
    return ops


def get_rocc_inline_asm(
    xcustom: str, func7: str, val1: SSAValue, val2: SSAValue
) -> llvm.InlineAsmOp:
    """
    This will emit a custom RoCC op with 2 source registers.
    As per the sources in:

    https://github.com/ucb-bar/gemmini-rocc-tests/blob/dev/include/gemmini.h

    #define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct)
        ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)

    https://github.com/IBM/rocc-software/blob/master/src/xcustom.h

    #define ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, func7)
    {
      asm volatile(
          ".insn r " STR(CAT(CUSTOM_, x)) ",
          " STR(0x3) ", " STR(func7) ", x0, %0, %1"
          :
          : "r"(rs1), "r"(rs2));
    }
    """
    return llvm.InlineAsmOp(
        (f".insn r CUSTOM_{xcustom}, 0x3, " f"{func7} ,x0, $0, $1"),
        "r, r",
        [val1, val2],
        has_side_effects=True,
    )
