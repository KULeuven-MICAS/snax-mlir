from typing import Sequence
from xdsl.ir import Operation
from xdsl.dialects.builtin import UnitAttr, StringAttr
from xdsl.dialects import test


from compiler.accelerators.accelerator import Accelerator
from compiler.dialects import accfg


class MatmulUnit(Accelerator):

    def convert_to_acc_ops(self, op: type[Operation]) -> Sequence[Operation]:
        # Left as an exercise to the reader (frontend stuff)
        pass

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        pass

    @staticmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        return [test.TestOp(attributes={
            "matmul_unit_await_op": UnitAttr(),
        })]

    @staticmethod
    def lower_acc_launch(launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        return [test.TestOp(attributes={
            "matmul_unit_launch_op": UnitAttr(),
            "accelerator": launch_op.accelerator,
        })]

    @staticmethod
    def lower_acc_setup(setup_op: accfg.SetupOp, acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        ops = []
        for name, val in setup_op.iter_params():
            ops.append(test.TestOp(
                operands=[val],
                attributes={
                    "matmul_unit_setup_op": UnitAttr(),
                    "accelerator": setup_op.accelerator,
                    "field": StringAttr(name),
                }
            ))
        return ops
