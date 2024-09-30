from collections.abc import Sequence
from math import prod

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
from compiler.accelerators.streamers.streamers import StreamerOpts
from compiler.dialects import accfg, snax_stream
from compiler.util.pack_bitlist import pack_bitlist

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.Reader,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=("n",),
            opts=(StreamerOpts.HasTranspose,),
        ),
        Streamer(  # B
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=("n",),
            opts=(StreamerOpts.HasTranspose,),
        ),
        Streamer(  # D8
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=("n",),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("r", "n", "n"),
            spatial_dims=("n",),
            opts=(StreamerOpts.HasChannelMask,),
        ),
        Streamer(  # D32
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=("n",),
        ),
    ],
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
            # csr0: min_int (i8) | max_int (i8) | out_zp (i8) | in_zp (i8)
            "csr0",
            # csr1: double_round (i8)
            "csr1",
            # 8 separate shift values
            *(f"shift_{i}" for i in range(2)),
            # 8 separate mult values
            *(f"mult_{i}" for i in range(8)),
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
                **{f"shift_{i}": addr_next + 6 + i for i in range(2)},
                **{f"mult_{i}": addr_next + 8 + i for i in range(8)},
                "temporal_loop_bound": addr_next + 16,
                "bypassSIMD": addr_next + 17,
            },
            {**streamer_launch, "launch_gemmx": addr_next + 18},
            addr_next + 18,
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
            csr0 = c0.result
            csr1 = c0.result
            shift_vals = (c0.result for _ in range(2))
            mult_vals = (c0.result for _ in range(8))

            # get zero points for gemm
            assert isinstance(qmac.zp_lhs, BlockArgument)
            zp_a = generic_op.inputs[qmac.zp_lhs.index]
            assert isinstance(qmac.zp_rhs, BlockArgument)
            zp_b = generic_op.inputs[qmac.zp_rhs.index]

            # bitwise and with 8b'11111111 to avoid the sign bits extending the 8-bit field
            # when bitlist packing
            ops_to_add.append(cst255 := arith.Constant.from_int_and_width(255, 32))
            ops_to_add.append(zp_a := arith.AndI(zp_a, cst255))
            ops_to_add.append(zp_b := arith.AndI(zp_b, cst255))

            bitlist = list(pack_bitlist((zp_a, zp_b), [0, 8]))
            ops_to_add.extend(bitlist)
            subtractions = bitlist[-1].results[0]

        elif isinstance(rescale := generic_op.body.block.first_op, kernel.RescaleOp):
            # extract and compute correct value for csr's based on kernel rescale op
            # set k to 1
            knm.insert(
                0, ((cst := arith.Constant.from_int_and_width(1, 32),), cst.result)
            )
            # simd
            bypassSIMD = c0.result
            subtractions = c0.result

            max_int = arith.Constant.from_int_and_width(rescale.max_int.value, i32)
            min_int = arith.Constant.from_int_and_width(rescale.min_int.value, i32)
            double_round = arith.Constant.from_int_and_width(
                rescale.double_round.value, i32
            )
            shift = arith.Constant.from_int_and_width(rescale.shift.value, i32)
            mult = arith.Constant.from_int_and_width(rescale.multiplier.value, i32)
            zp_in = arith.Constant.from_int_and_width(rescale.input_zp.value, i32)
            zp_out = arith.Constant.from_int_and_width(rescale.output_zp.value, i32)
            ops_to_add.extend(
                [max_int, min_int, double_round, shift, mult, zp_in, zp_out]
            )

            # force values that can be negative to 8 bits
            cst255 = arith.Constant.from_int_and_width(255, 32)
            max_int = arith.AndI(max_int, cst255)
            min_int = arith.AndI(min_int, cst255)
            zp_in = arith.AndI(zp_in, cst255)
            zp_out = arith.AndI(zp_out, cst255)
            ops_to_add.extend([cst255, max_int, min_int, zp_in, zp_out])

            # bitpacking
            ops_to_add.extend(
                pack_bitlist([min_int, max_int, zp_out, zp_in], [24, 16, 8, 0])
            )
            csr0 = ops_to_add[-1].results[0].op.results[0]
            csr1 = double_round.result

            shift_bitlist = list(pack_bitlist((shift,) * 4, (24, 16, 8, 0)))
            ops_to_add.extend(shift_bitlist)

            shift_vals = (shift_bitlist[-1].results[0] for _ in range(2))
            mult_vals = (mult.result for _ in range(8))

            loop_bound = prod(x.data for x in op.stride_patterns.data[0].upper_bounds)
            loop_bound = arith.Constant.from_int_and_width(loop_bound, i32)
            ops_to_add.append(loop_bound)

        else:
            raise NotImplementedError()

        return [
            *streamer_setup_vals,
            *knm,
            ([c0, c1, *ops_to_add], subtractions),  # subtractions
            ([], csr0),  # csr0
            ([], csr1),  # csr1
            *(([], x) for x in shift_vals),
            *(([], x) for x in mult_vals),
            ([], loop_bound.result),  # temporal_loop_bound
            ([], bypassSIMD),  # bypassSIMD
        ]
