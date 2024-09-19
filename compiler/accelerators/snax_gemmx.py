from collections.abc import Sequence

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Operation, SSAValue

import compiler.dialects.kernel as kernel
from compiler.accelerators.dispatching import DispatchTemplate, SupportedKernel
from compiler.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier4,
    SNAXStreamer,
)
from compiler.accelerators.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from compiler.dialects import accfg, snax_stream

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.Reader,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=("n", "i", "n"),
        ),
        Streamer(  # B
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("n", "n", "i"),
        ),
        Streamer(  # D8
            StreamerType.Writer,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("i", "n", "n"),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("i", "n", "n"),
        ),
        Streamer(  # D32
            StreamerType.ReaderWriter,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("i", "n", "n"),
        ),
    ],
    separate_bounds=True,
)


class SNAXGEMMXAccelerator(
    SNAXAccelerator, SNAXStreamer, DispatchTemplate, SNAXPollingBarrier4
):
    """
    Accelerator Interface class for SNAX GEMMX accelerator
    """

    name = "snax_gemmx"

    supported_kernels = (
        SupportedKernel(kernel.QMacOp, (i8, i8, i32, i32, i32)),
        SupportedKernel(kernel.RescaleOp, (i32, i8)),
    )

    def __init__(
        self, streamer_config: StreamerConfiguration = default_streamer
    ) -> None:
        super().__init__(streamer_config)

        self.fields = (
            *self.streamer_setup_fields,
            "K",
            "N",
            "M",
            # subtractions: zp_b (i8) | zp_a (i8)
            "subtractions",
            # csr0: max_int (i8) | shift (i8) | out_zp (i8) | in_zp (i8)
            "csr0",
            # csr1: double_round (i8) | min_int (i8)
            "csr1",
            # csr2: multiplier (i32)
            "csr2",
            "temporal_loop_bound",
            "bypassSIMD",
        )

        self.launch_fields = (*self.streamer_launch_fields, "launch_gemmx")

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX GEMMX accelerator op
        """

        # base address:
        base_addr = 0x3C0

        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
                "K": addr_next + 0,
                "N": addr_next + 1,
                "M": addr_next + 2,
                "subtractions": addr_next + 3,
                "csr0": addr_next + 4,
                "csr1": addr_next + 5,
                "csr2": addr_next + 6,
                "temporal_loop_bound": addr_next + 7,
                "bypassSIMD": addr_next + 8,
            },
            {**streamer_launch, "launch_gemm": addr_next + 9},
            addr_next + 9,
        )
        op.attributes["streamer_config"] = self.streamer_config
        return op

    def convert_to_acc_ops(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[Operation]:
        args = self._generate_setup_vals(op)

        ops_to_insert = []
        # insert ops to calculate arguments
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.Constant(builtin.IntegerAttr.from_int_and_width(1, 5)),
            token := accfg.LaunchOp(
                [launch_val, launch_val], self.launch_fields, setup
            ),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """

        raise NotImplementedError()
