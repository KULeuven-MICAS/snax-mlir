from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg, memref
from xdsl.dialects.builtin import i64
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import AffineMap
from xdsl.utils.hints import isa

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
        Streamer.from_dim(StreamerType.Reader, 1, 1),
        Streamer.from_dim(StreamerType.Reader, 1, 1),
        Streamer.from_dim(StreamerType.Writer, 1, 1),
    ]
)


class SNAXAluAccelerator(
    SNAXAccelerator, SNAXPollingBarrier3, SNAXStreamer, DispatchTemplate
):
    """
    Accelerator interface class for the SNAX Alu accelerator.
    """

    name = "snax_alu"

    supported_kernels = (
        SupportedKernel(kernel.AddOp, [i64, i64, i64]),
        SupportedKernel(kernel.MulOp, [i64, i64, i64]),
    )

    def __init__(
        self, streamer_config: StreamerConfiguration = default_streamer
    ) -> None:
        super().__init__(streamer_config)

        self.fields = (*self.streamer_setup_fields, "alu_mode", "loop_bound_alu")
        self.launch_fields = (*self.streamer_launch_fields, "launch_alu")

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        # linalg.generic lowering is stil hardcoded, but kept until
        # lowering from linalg -> snax_stream is complete
        if isinstance(op, linalg.GenericOp):
            args = self._generate_setup_vals(op)
        elif isinstance(op, snax_stream.StreamingRegionOp):
            args = self._generate_stream_setup_vals(op)
        else:
            return []

        ops_to_insert: Sequence[Operation] = []
        for new_ops, _ in args:
            ops_to_insert.extend(new_ops)

        return [
            *ops_to_insert,
            setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
            launch_val := arith.ConstantOp(
                builtin.IntegerAttr.from_int_and_width(1, 5)
            ),
            token := accfg.LaunchOp(
                [launch_val, launch_val], self.launch_fields, setup
            ),
            accfg.AwaitOp(token),
        ]

    def _generate_setup_vals(
        self, op: linalg.GenericOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        a, b, o = op.operands

        c0_index = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
        dim_0 = memref.DimOp.from_source_and_index(a, c0_index)
        design_time_parallelism = arith.ConstantOp.from_int_and_width(
            4, builtin.IndexType()
        )
        loop_bound = arith.DivUIOp(dim_0, design_time_parallelism)
        loop_bound_i32 = arith.IndexCastOp(loop_bound, builtin.i32)
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        c8 = arith.ConstantOp.from_int_and_width(8, 32)
        c32 = arith.ConstantOp.from_int_and_width(32, 32)

        ptrs: Sequence[tuple[Sequence[Operation], SSAValue]] = []

        for ref in (a, b, o):
            assert isa(ref.type, builtin.MemRefType[builtin.IntegerType])
            ptrs.append(
                (
                    [
                        ptr := memref.ExtractAlignedPointerAsIndexOp.get(ref),
                        metadata := memref.ExtractStridedMetaDataOp(ref),
                        el_bytes := arith.ConstantOp.from_int_and_width(
                            ref.type.element_type.size, builtin.IndexType()
                        ),
                        byte_offset := arith.MuliOp(metadata.offset, el_bytes),
                        ptr_plus_byte_offset := arith.AddiOp(
                            ptr, byte_offset, builtin.IndexType()
                        ),
                        ptr_i32 := arith.IndexCastOp(ptr_plus_byte_offset, builtin.i32),
                    ],
                    ptr_i32.result,
                )
            )

        return [
            # streamer 1
            # base pointer
            (ptrs[0]),
            ([c0], c0.result),
            # spatial stride
            ([c8], c8.result),
            # loop bound streamer
            (
                [c0_index, dim_0, design_time_parallelism, loop_bound, loop_bound_i32],
                loop_bound_i32.result,
            ),
            # temporal strides streamers
            ([c32], c32.result),
            # streamer 2 base ptr
            (ptrs[1]),
            ([], c0.result),
            # spatial stride
            ([], c8.result),
            # loop bound
            ([], loop_bound_i32.result),
            # temporal stride
            ([], c32.result),
            # streamer 3 base ptr
            (ptrs[2]),
            ([], c0.result),
            # spatial stride
            ([], c8.result),
            # loop bound
            ([], loop_bound_i32.result),
            # temporal stride
            ([], c32.result),
            # alu mode
            ([], c0.result),
            # alu iterations
            ([], loop_bound_i32.result),
        ]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        loop_bound = arith.ConstantOp.from_int_and_width(
            op.stride_patterns.data[0].upper_bounds.data[0], 32
        )

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

    @staticmethod
    def get_template(op: dart.StreamingRegionOpBase):
        template = [AffineMap.from_callable(lambda y: (y,))] * 3
        template_bounds = (4,)
        return Template(TemplatePattern(template_bounds, tp) for tp in template)
