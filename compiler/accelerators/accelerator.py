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
