from collections.abc import Sequence

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i32
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineMap

import snaxc.dialects.kernel as kernel
from snaxc.accelerators.dispatching import DispatchTemplate, SupportedKernel
from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier3,
    SNAXStreamer,
)
from snaxc.accelerators.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.dialects import accfg, dart, snax_stream
from snaxc.ir.dart.access_pattern import Template, TemplatePattern

default_streamer = StreamerConfiguration(
    [
        Streamer(StreamerType.Reader, ["n", "n", "n", "n", "n"], [8]),
        Streamer(StreamerType.Writer, ["n", "n", "n"], [8]),
    ]
)


class SNAXXDMAAccelerator(SNAXAccelerator, SNAXPollingBarrier3, SNAXStreamer, DispatchTemplate):
    """
    Accelerator interface class for the SNAX XDMA.
    """

    name = "snax_xdma"

    supported_kernels = (SupportedKernel(kernel.AddOp, [i32, i32, i32]),)

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
        super().__init__(streamer_config)

        self.fields = (*self.streamer_setup_fields, "alu_mode", "loop_bound_alu")
        self.launch_fields = (*self.streamer_launch_fields, "launch_alu")

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        if isinstance(op, snax_stream.StreamingRegionOp):
            args = self._generate_stream_setup_vals(op)
        else:
            return []

        ops_to_insert: Sequence[Operation] = []
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(1, 5)),
            token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
            accfg.AwaitOp(token),
        ]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        loop_bound = arith.ConstantOp.from_int_and_width(op.stride_patterns.data[0].upper_bounds.data[0], 32)

        return [
            *self._generate_streamer_setup_vals(op),
            ([c0], c0.result),
            ([loop_bound], loop_bound.result),
        ]

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        # base address:
        base_addr = 0x3C0

        # streamer setup addresses
        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        # streamer launch addresses
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
                "alu_mode": addr_next + 0,
                "loop_bound_alu": addr_next + 1,
            },
            {**streamer_launch, "launch_alu": addr_next + 2},
            addr_next + 3,
        )

        # add snax streamer interface
        op.attributes["streamer_config"] = self.streamer_config

        return op

    def get_template(self, op: dart.StreamingRegionOpBase):
        template = [AffineMap.from_callable(lambda y: (y,))] * 2
        template_bounds = (8,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)
