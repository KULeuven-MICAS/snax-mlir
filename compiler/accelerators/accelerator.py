from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation, SSAValue

from compiler.dialects import acc


class Accelerator(ABC):
    """
    Interface to lower to and from acc2 dialect.
    """

    name: str
    fields: tuple[str]

    @abstractmethod
    def generate_setup_vals(
        self, op: Operation
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """
        pass

    @abstractmethod
    def generate_acc_op(self) -> acc.AcceleratorOp:
        """
        Return an accelerator op:

        "acc2.accelerator"() <{
            name            = @name_of_the_accelerator,
            fields          = {field_1=address_1, field_2=address2},
            launch_addr     = launch_address,
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
