from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation

from compiler.dialects import acc


class Accelerator(ABC):
    """
    Interface to lower to and from acc2 dialect.
    """

    name: str

    @abstractmethod
    def convert_to_acc_ops(self, op: type[Operation]) -> Sequence[Operation]:
        """
        Lowers the operation op to a sequence of acc_ops.
        acc_ops are:
            - *.op that generates SSAValues consumed by acc2.setup
            - acc2.setup
            - acc2.launch
            - acc2.await
        These ops can further be lowered by specific instances of the
        Accelerator interface
        """
        pass

    @abstractmethod
    def generate_acc_op(self) -> acc.AcceleratorOp:
        """
        Return an accelerator op:

        "acc2.accelerator"() <{
            name            = @name_of_the_accelerator,
            fields          = {field_1=address_1, field_2=address2},
            launch_fields   = {launch_field_1=address_1,
            barrier         = barrier_address,
        }> : () -> ()
        """
        pass

    @staticmethod
    @abstractmethod
    def lower_acc_await(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        """
        Based on the acc2.accelerator op, return the necessary sequence of
        lower-level operations to perform
        asynchronous await on the accelerator.
        """
        pass

    @staticmethod
    @abstractmethod
    def lower_acc_launch(acc_op: acc.AcceleratorOp) -> Sequence[Operation]:
        """
        Based on the acc2.accelerator op, return the necessary sequence of
        lower-level operations to perform an
        asynchronous launch of the accelerator.
        """
        pass

    @staticmethod
    @abstractmethod
    def lower_acc_setup(
        setup_op: acc.SetupOp, acc_op: acc.AcceleratorOp
    ) -> Sequence[Operation]:
        """
        Based on the acc2.accelerator op and the acc.SetupOp,
        return the necessary sequence of lower-level operations to perform
        accelerator configuration.
        """
        pass
