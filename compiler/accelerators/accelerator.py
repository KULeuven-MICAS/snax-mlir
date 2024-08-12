from abc import ABC, abstractmethod
from collections.abc import Sequence

from xdsl.ir import Operation

from compiler.dialects import accfg


class Accelerator(ABC):
    """
    Interface to lower to and from accfg dialect.
    """

    name: str

    @abstractmethod
    def convert_to_acc_ops(self, op: type[Operation]) -> Sequence[Operation]:
        """
        Lowers the operation op to a sequence of acc_ops.
        acc_ops are:
            - *.op that generates SSAValues consumed by accfg.setup
            - accfg.setup
            - accfg.launch
            - accfg.await
        These ops can further be lowered by specific instances of the
        Accelerator interface
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return an accelerator op:

        "accfg.accelerator"() <{
            name            = @name_of_the_accelerator,
            fields          = {field_1=address_1, field_2=address2},
            launch_fields   = {launch_field_1=address_1,
            barrier         = barrier_address,
        }> : () -> ()
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def lower_acc_await(acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        """
        Based on the accfg.accelerator op, return the necessary sequence of
        lower-level operations to perform
        asynchronous await on the accelerator.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def lower_acc_launch(
        launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
        """
        Based on the accfg.accelerator op, return the necessary sequence of
        lower-level operations to perform an
        asynchronous launch of the accelerator.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def lower_acc_setup(
        setup_op: accfg.SetupOp, acc_op: accfg.AcceleratorOp
    ) -> Sequence[Operation]:
        """
        Based on the accfg.accelerator op and the accfg.SetupOp,
        return the necessary sequence of lower-level operations to perform
        accelerator configuration.
        """
        raise NotImplementedError()
