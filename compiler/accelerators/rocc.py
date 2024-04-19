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
    def lower_acc_launch(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        """
        There are no accelerator launch operations for RoCC
        """
        return []

    @staticmethod
    def lower_acc_setup(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        """
        #define ROCC_INSTRUCTION_RS1_RS2(x, rs1, rs2, funct)
              ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, funct)
                #define ROCC_INSTRUCTION_0_R_R(x, rs1, rs2, func7)
                  {
                    asm volatile(
                        ".insn r " STR(CAT(CUSTOM_, x)) ",
                        " STR(0x3) ", " STR(func7) ", x0, %0, %1"
                        :
                        : "r"(rs1), "r"(rs2));
                  }
        """
        xcustom_acc = 3  # hardcoded to 3 for now
        setup_dict = dict(setup_op.iter_params())

        # Assert that pairs exist for each item in the setup op
        # Starting from rs1 ops
        for name in [name for name in acc_op.field_names() if name.endswith(".rs1")]:
            if name in setup_dict:
                assert name[:-4:] + ".rs2" in setup_dict
        # Starting from rs2 ops
        for name in [name for name in acc_op.field_names() if name.endswith(".rs2")]:
            if name in setup_dict:
                assert name[:-4:] + ".rs1" in setup_dict

        # Create a dictionary that contains the two vals associated
        # to each single RoCC instruction
        vals: dict[str, list[SSAValue]] = {}
        for field, val in setup_op.iter_params():
            # Strip .rs1 or .rs2 off of the name
            vals.setdefault(field[:-4], []).append(val)

        # Create the sequence of all operations that need to be emitted
        ops: Sequence[Operation] = []
        for name, func7 in [
            (name, func7.value.data)
            for name, func7 in acc_op.field_items()
            if name.endswith(".rs1")
        ]:
            ops.extend(
                [
                    llvm.InlineAsmOp(
                        (
                            f".insn r {'CUSTOM_'+str(xcustom_acc)}, 0x3, "
                            f"{str(func7)} ,x0, $0, $1"
                        ),
                        "r, r",
                        [vals[name[:-4]][0], vals[name[:-4]][1]],
                        has_side_effects=True,
                    ),
                ]
            )
        return ops
