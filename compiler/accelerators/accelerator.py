from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation, SSAValue

from compiler.dialects import acc


class AcceleratorInfo(ABC):
    @abstractmethod
    def generate_setup_vals(
        self, op: Operation
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        pass

    @abstractmethod
    def generate_acc_op(self) -> acc.AcceleratorOp:
        pass

    @abstractmethod
    @staticmethod
    def lower_acc_barrier(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        pass

    @abstractmethod
    @staticmethod
    def lower_acc_launch(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        pass

    @abstractmethod
    @staticmethod
    def lower_setup_op(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        pass
