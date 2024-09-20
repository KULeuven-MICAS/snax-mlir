from collections.abc import Sequence

from xdsl.dialects import arith, builtin, memref_stream
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import BlockArgument, Operation, SSAValue

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
from compiler.util.pack_bitlist import pack_bitlist

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.ReaderTranspose,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=("n", "i", "n"),
        ),
        Streamer(  # B
            StreamerType.ReaderTranspose,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("n", "n", "i"),
        ),
        Streamer(  # D8
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=("i", "n", "n"),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("r", "n", "n"),
            spatial_dims=("i", "n", "n"),
        ),
        Streamer(  # D32
            StreamerType.ReaderWriter,
            temporal_dims=("r", "n", "n"),
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
            {**streamer_launch, "launch_gemmx": addr_next + 9},
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

        c0 = arith.Constant.from_int_and_width(0, 32)
        c1 = arith.Constant.from_int_and_width(1, 32)
        knm: list = [
            (((cst := arith.Constant.from_int_and_width(val.data, 32)),), cst.result)
            for val in op.stride_patterns.data[0].upper_bounds
        ]

        streamer_setup_vals = list(self._generate_streamer_setup_vals(op))

        ops_to_add: list[Operation] = []

        assert isinstance(generic_op := op.body.block.first_op, memref_stream.GenericOp)

        if isinstance(qmac := generic_op.body.block.first_op, kernel.QMacOp):
            # gemm
            # bypass simd and set all related values to 0
            bypassSIMD = c1.result  # bypass simd
            loop_bound = c0
            csr0 = c0
            csr1 = c0
            csr2 = c0

            # get zero points for gemm
            assert isinstance(qmac.zp_lhs, BlockArgument)
            zp_a = generic_op.inputs[qmac.zp_lhs.index]
            assert isinstance(qmac.zp_rhs, BlockArgument)
            zp_b = generic_op.inputs[qmac.zp_rhs.index]

            # bitwise and with 8b'11111111 to don't fuck up bitpacking in case of negative values
            ops_to_add.append(cst255 := arith.Constant.from_int_and_width(255, 32))
            ops_to_add.append(zp_a := arith.AndI(zp_a, cst255))
            ops_to_add.append(zp_b := arith.AndI(zp_b, cst255))

            bitlist = list(pack_bitlist((zp_a, zp_b), [0, 8]))
            ops_to_add.extend(bitlist)
            subtractions = bitlist[-1].results[0]

        else:
            # simd
            raise NotImplementedError()

        return [
            *streamer_setup_vals,
            *knm,
            ([c0, c1, *ops_to_add], subtractions),  # subtractions
            ([], csr0.result),  # csr0
            ([], csr1.result),  # csr1
            ([], csr2.result),  # csr2
            ([], loop_bound.result),  # temporal_loop_bound
            ([], bypassSIMD),  # bypassSIMD
        ]
