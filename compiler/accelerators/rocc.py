from abc import ABC
from collections.abc import Sequence

from xdsl.dialects import llvm
from xdsl.ir import Operation, SSAValue

from compiler.accelerators.accelerator import Accelerator
from compiler.dialects import acc


class RoCCAccelerator(Accelerator, ABC):
    """
    Abstract base class for extending AcceleratorInterfaces
    with common RoCC lowerings.
    """

    @staticmethod
    def lower_acc_await(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        """
        There are no accelerator barrier operations for RoCC
        """
        return []

    @staticmethod
    def lower_acc_launch(
        launch_op: acc.LaunchOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        xcustom_acc = 3  # hardcoded to 3 for now
        vals = create_launch_pairs(launch_op, acc_op)
        # Create the sequence of all operations that need to be emitted
        ops: Sequence[Operation] = []
        for name, func7 in [
            (name, func7.value.data)
            for name, func7 in acc_op.launch_field_items()
            if name.endswith(".rs1")
        ]:
            ops.extend(
                [
                    get_rocc_inline_asm(
                        str(xcustom_acc),
                        str(func7),
                        vals[name[:-4]][0],
                        vals[name[:-4]][1],
                    ),
                ]
            )
        return ops

    @staticmethod
    def lower_acc_setup(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        xcustom_acc = 3  # hardcoded to 3 for now
        vals = create_setup_pairs(setup_op, acc_op)
        # Create the sequence of all operations that need to be emitted
        ops: Sequence[Operation] = []
        for name, func7 in [
            (name, func7.value.data)
            for name, func7 in acc_op.field_items()
            if name.endswith(".rs1")
        ]:
            ops.extend(
                [
                    get_rocc_inline_asm(
                        str(xcustom_acc),
                        str(func7),
                        vals[name[:-4]][0],
                        vals[name[:-4]][1],
                    ),
                ]
            )
        return ops


def assert_pairs(field_dict, field_names):
    """
    Assert that pairs of rs1 and rs2 exist for each item in the fields
    """
    # Starting from rs1 ops
    for name in [name for name in field_names if name.endswith(".rs1")]:
        if name in field_dict:
            assert name[:-4:] + ".rs2" in field_dict
    # Starting from rs2 ops
    for name in [name for name in field_names if name.endswith(".rs2")]:
        if name in field_dict:
            assert name[:-4:] + ".rs1" in field_dict


def create_setup_pairs(fields_op: acc.LaunchOp, acc_op: acc.AcceleratorOp):
    return create_pairs(fields_op, acc_op.field_names())


def create_launch_pairs(fields_op: acc.LaunchOp, acc_op: acc.AcceleratorOp):
    return create_pairs(fields_op, acc_op.launch_field_names())


def create_pairs(fields_op: acc.LaunchOp, field_names):
    launch_dict = dict(fields_op.iter_params())
    assert_pairs(launch_dict, field_names)
    # Create a dictionary that contains the two vals associated
    # to each single RoCC instruction
    vals: dict[str, list[SSAValue]] = {}
    for field, val in fields_op.iter_params():
        # Strip .rs1 or .rs2 off of the name
        vals.setdefault(field[:-4], []).append(val)
    return vals


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
