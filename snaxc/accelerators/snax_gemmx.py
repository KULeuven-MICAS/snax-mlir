from collections.abc import Sequence
from math import prod

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import BlockArgument, Operation, OpResult, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap

from snaxc.accelerators.dispatching import DispatchTemplate, SupportedKernel
from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier4,
    SNAXStreamer,
)
from snaxc.accelerators.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.accelerators.streamers.streamers import StreamerOpts
from snaxc.dialects import accfg, dart, kernel, snax_stream
from snaxc.ir.dart.access_pattern import Template, TemplatePattern
from snaxc.util.pack_bitlist import pack_bitlist

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.Reader,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=(8,),
            opts=(StreamerOpts.HasTranspose, StreamerOpts.HasAddressRemap),
        ),
        Streamer(  # B
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=(8,),
            opts=(StreamerOpts.HasTranspose, StreamerOpts.HasAddressRemap),
        ),
        Streamer(  # D8
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8,),
            opts=(StreamerOpts.HasAddressRemap,),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8, 4),
            opts=(
                StreamerOpts.HasChannelMask,
                StreamerOpts.HasAddressRemap,
                StreamerOpts.HasBroadcast,
            ),
        ),
        Streamer(  # D32
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8, 4),
            opts=(StreamerOpts.HasAddressRemap,),
        ),
    ],
)


class SNAXGEMMXAccelerator(SNAXAccelerator, SNAXStreamer, DispatchTemplate, SNAXPollingBarrier4):
    """
    Accelerator Interface class for SNAX GEMMX accelerator
    """

    name = "snax_gemmx"

    supported_kernels = (
        SupportedKernel(kernel.QMacOp, (i8, i8, i32, i32, i32)),
        SupportedKernel(kernel.MacOp, (i8, i8, i32)),
        SupportedKernel(kernel.AddOp, (i32, i32, i32)),
        SupportedKernel(kernel.RescaleOp, (i32, i8)),
    )

    def __init__(self, streamer_config: StreamerConfiguration = default_streamer) -> None:
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

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        if not isinstance(op, snax_stream.StreamingRegionOp):
            return []
        else:
            args = self._generate_setup_vals(op)

            ops_to_insert: Sequence[Operation] = []
            # insert ops to calculate arguments
            for new_ops, _ in args:
                ops_to_insert.extend(new_ops)

            return [
                *ops_to_insert,
                setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
                launch_val := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(1, 5)),
                token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
                accfg.AwaitOp(token),
            ]

    def _generate_setup_vals(self, op: snax_stream.StreamingRegionOp) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        """
        Produce a `Sequence[Operation], SSAValue` tuple
        for each field that contains:

        - a list of operations that calculate the field value
        - a reference to the SSAValue containing the calculated field value
        """

        c0 = arith.ConstantOp.from_int_and_width(0, 32)
        c1 = arith.ConstantOp.from_int_and_width(1, 32)

        streamer_setup_vals = list(self._generate_streamer_setup_vals(op))

        ops_to_add: list[Operation] = []

        assert isinstance(generic_op := op.body.block.first_op, dart.GenericOp)
        assert isinstance(yield_op := op.body.block.last_op, dart.YieldOp)

        if isinstance(qmac := generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            if yield_op.arguments[0].type == dart.StreamType(builtin.IntegerType(8)):
                i8_out = True
                last_pattern = op.stride_patterns.data[2]
            else:
                i8_out = False
                last_pattern = op.stride_patterns.data[-1]
            # compute knm: fix n = 1
            n = 1
            # count the number of non-reducing bounds (output stride != 0)
            m = (
                prod(
                    bound.data
                    for bound, stride in zip(
                        last_pattern.upper_bounds,
                        last_pattern.temporal_strides,
                    )
                    if stride.data != 0
                )
                // n
            )
            k = prod(x.data for x in op.stride_patterns.data[0].upper_bounds) // m

            # gemm
            # bypass simd and set all related values to 0
            if i8_out:
                bypassSIMD = c0.result  # bypass simd
            else:
                bypassSIMD = c1.result  # bypass simd
            if i8_out:
                # check if there is a rescale op:
                region_yield = op.body.block.last_op
                assert isinstance(region_yield, dart.YieldOp)
                if isinstance(region_yield.prev_op, dart.GenericOp) and isinstance(
                    rescale_op := region_yield.prev_op.body.block.first_op, kernel.RescaleOp
                ):
                    max_int_val = rescale_op.max_int.value.data
                    min_int_val = rescale_op.min_int.value.data
                    double_round_val = rescale_op.double_round.value.data
                    shift_val = rescale_op.shift.value.data
                    mult_val = rescale_op.multiplier.value.data
                    zp_in_val = rescale_op.input_zp.value.data
                    zp_out_val = rescale_op.output_zp.value.data
                else:
                    max_int_val = 127
                    min_int_val = -128
                    double_round_val = 0
                    shift_val = 9
                    mult_val = 1
                    zp_in_val = 0
                    zp_out_val = 0

                m_val = arith.ConstantOp.from_int_and_width(m, 32)
                ops_to_add.append(m_val)
                loop_bound = m_val

                max_int = arith.ConstantOp.from_int_and_width(max_int_val, i32)
                min_int = arith.ConstantOp.from_int_and_width(min_int_val, i32)
                double_round = arith.ConstantOp.from_int_and_width(double_round_val, i32)
                shift = arith.ConstantOp.from_int_and_width(shift_val, i32)
                mult = arith.ConstantOp.from_int_and_width(mult_val, i32)
                zp_in = arith.ConstantOp.from_int_and_width(zp_in_val, i32)
                zp_out = arith.ConstantOp.from_int_and_width(zp_out_val, i32)
                ops_to_add.extend([max_int, min_int, double_round, shift, mult, zp_in, zp_out])

                # force values that can be negative to 8 bits
                cst255 = arith.ConstantOp.from_int_and_width(255, 32)
                max_int = arith.AndIOp(max_int, cst255)
                min_int = arith.AndIOp(min_int, cst255)
                zp_in = arith.AndIOp(zp_in, cst255)
                zp_out = arith.AndIOp(zp_out, cst255)
                ops_to_add.extend([cst255, max_int, min_int, zp_in, zp_out])

                # bitpacking
                ops_to_add.extend(pack_bitlist([min_int, max_int, zp_out, zp_in], [24, 16, 8, 0]))
                csr0 = ops_to_add[-1].results[0].op.results[0]
                csr1 = double_round.result

                shift_bitlist = list(pack_bitlist((shift,) * 4, (24, 16, 8, 0)))
                ops_to_add.extend(shift_bitlist)

                shift_vals = (shift_bitlist[-1].results[0] for _ in range(2))
                mult_vals = (mult.result for _ in range(8))

            else:
                loop_bound = c0
                csr0 = c0.result
                csr1 = c0.result
                shift_vals = (c0.result for _ in range(2))
                mult_vals = (c1.result for _ in range(8))

            if isinstance(qmac, kernel.QMacOp):
                # get zero points for gemm
                assert isinstance(qmac.zp_lhs, BlockArgument)
                zp_a = generic_op.inputs[qmac.zp_lhs.index]
                assert isinstance(qmac.zp_rhs, BlockArgument)
                zp_b = generic_op.inputs[qmac.zp_rhs.index]
            else:
                zp_a = c0
                zp_b = c0

            # bitwise and with 8b'11111111 to avoid the sign bits extending the 8-bit field
            # when bitlist packing
            ops_to_add.append(cst255 := arith.ConstantOp.from_int_and_width(255, 32))
            ops_to_add.append(zp_a := arith.AndIOp(zp_a, cst255))
            ops_to_add.append(zp_b := arith.AndIOp(zp_b, cst255))

            bitlist = list(pack_bitlist((zp_a, zp_b), [0, 8]))
            ops_to_add.extend(bitlist)
            subtractions = bitlist[-1].results[0]

        elif isinstance(rescale := generic_op.body.block.first_op, kernel.RescaleOp):
            # set k and n to 1
            k = 1
            n = 1
            m = prod(x.data for x in op.stride_patterns.data[0].upper_bounds)

            # extract and compute correct value for csr's based on kernel rescale op
            # simd
            bypassSIMD = c0.result
            subtractions = c0.result

            max_int = arith.ConstantOp.from_int_and_width(rescale.max_int.value, i32)
            min_int = arith.ConstantOp.from_int_and_width(rescale.min_int.value, i32)
            double_round = arith.ConstantOp.from_int_and_width(rescale.double_round.value, i32)
            shift = arith.ConstantOp.from_int_and_width(rescale.shift.value, i32)
            mult = arith.ConstantOp.from_int_and_width(rescale.multiplier.value, i32)
            zp_in = arith.ConstantOp.from_int_and_width(rescale.input_zp.value, i32)
            zp_out = arith.ConstantOp.from_int_and_width(rescale.output_zp.value, i32)
            ops_to_add.extend([max_int, min_int, double_round, shift, mult, zp_in, zp_out])

            # force values that can be negative to 8 bits
            cst255 = arith.ConstantOp.from_int_and_width(255, 32)
            max_int = arith.AndIOp(max_int, cst255)
            min_int = arith.AndIOp(min_int, cst255)
            zp_in = arith.AndIOp(zp_in, cst255)
            zp_out = arith.AndIOp(zp_out, cst255)
            ops_to_add.extend([cst255, max_int, min_int, zp_in, zp_out])

            # bitpacking
            ops_to_add.extend(pack_bitlist([min_int, max_int, zp_out, zp_in], [24, 16, 8, 0]))
            csr0 = ops_to_add[-1].results[0].op.results[0]
            csr1 = double_round.result

            shift_bitlist = list(pack_bitlist((shift,) * 4, (24, 16, 8, 0)))
            ops_to_add.extend(shift_bitlist)

            shift_vals = (shift_bitlist[-1].results[0] for _ in range(2))
            mult_vals = (mult.result for _ in range(8))

            loop_bound = prod(x.data for x in op.stride_patterns.data[0].upper_bounds)
            loop_bound = arith.ConstantOp.from_int_and_width(loop_bound, i32)
            ops_to_add.append(loop_bound)

        else:
            raise NotImplementedError()

        knm: list[tuple[tuple[Operation], OpResult]] = [
            (((cst := arith.ConstantOp.from_int_and_width(val, 32)),), cst.result) for val in (k, n, m)
        ]

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

    @staticmethod
    def get_template(op: dart.StreamingRegionOpBase) -> Template:
        assert isinstance(generic_op := op.body.block.first_op, dart.GenericOp)
        if isinstance(generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            # matmul
            m, n, k = (AffineDimExpr(i) for i in range(3))
            template = [
                AffineMap(3, 0, (m, k)),
                AffineMap(3, 0, (k, n)),
                AffineMap(3, 0, (m, n)),
            ]
            template_bounds = (8, 8, 8)

            if isinstance(generic_op.next_op, dart.GenericOp):
                generic_op = generic_op.next_op
                if isinstance(generic_op.body.block.first_op, kernel.AddOp):
                    # gemm, add c pattern that is equal to output pattern
                    template += [template[-1]]
                elif isinstance(generic_op.body.block.first_op, kernel.RescaleOp):
                    # same template
                    pass
                else:
                    raise RuntimeError("unsupported kernel")
            if isinstance(generic_op.next_op, dart.GenericOp):
                generic_op = generic_op.next_op
                if isinstance(generic_op.body.block.first_op, kernel.RescaleOp):
                    # same template
                    pass
                else:
                    raise RuntimeError("unsupported kernel")
        else:
            # rescale only function of gemmx
            m, k = (AffineDimExpr(i) for i in range(2))
            template = [
                AffineMap(2, 0, (m, k)),
                AffineMap(2, 0, (m, k)),
            ]
            template_bounds = (8, 8)

        if not isinstance(generic_op.next_op, dart.YieldOp):
            raise RuntimeError("unsupported kernel")

        return Template(TemplatePattern(template_bounds, tp) for tp in template)
