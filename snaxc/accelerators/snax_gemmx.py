from collections.abc import Sequence
from math import ceil, prod
from typing import Self, cast

from xdsl.dialects import arith, builtin, llvm
from xdsl.dialects.builtin import DenseArrayBase, i8, i32
from xdsl.ir import Attribute, BlockArgument, Operation, OpResult, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.parser import IntegerType
from xdsl.utils.hints import isa

from snaxc.accelerators.configurable_accelerator import ConfigurableAccelerator
from snaxc.accelerators.dispatching import DispatchTemplate, SupportedKernel
from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier4,
    SNAXStreamer,
)
from snaxc.accelerators.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasChannelMask,
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.accelerators.streamers.extensions import TransposeExtension
from snaxc.accelerators.streamers.streamers import StreamerOpts
from snaxc.dialects import accfg, dart, kernel, snax_stream
from snaxc.ir.dart.access_pattern import Template, TemplatePattern
from snaxc.tools.configs import AcceleratorConfig, GemmxConfig
from snaxc.util.pack_bitlist import pack_bitlist

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.Reader,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=(8,),
            opts=(TransposeExtension(), HasAddressRemap()),
        ),
        Streamer(  # B
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=(8,),
            opts=(TransposeExtension(), HasAddressRemap()),
        ),
        Streamer(  # D8
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8,),
            opts=(HasAddressRemap(),),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8, 4),
            opts=(
                HasChannelMask(),
                HasAddressRemap(),
                HasBroadcast(),
            ),
        ),
        Streamer(  # D32
            StreamerType.Writer,
            temporal_dims=("r", "n", "n"),
            spatial_dims=(8, 4),
            opts=(HasAddressRemap(),),
        ),
    ],
)


class SNAXGEMMXAccelerator(
    SNAXAccelerator, SNAXStreamer, DispatchTemplate, SNAXPollingBarrier4, ConfigurableAccelerator
):
    """
    Accelerator Interface class for SNAX GEMMX accelerator
    """

    name = "snax_gemmx"
    m: int = 8
    n: int = 8
    k: int = 8

    serializer_ratio: int

    supported_kernels = (
        SupportedKernel(kernel.QMacOp, (i8, i8, i32, i32, i32)),
        SupportedKernel(kernel.MacOp, (i8, i8, i32)),
        SupportedKernel(kernel.AddOp, (i32, i32, i32)),
        SupportedKernel(kernel.RescaleOp, (i32, i8)),
    )

    def __init__(
        self,
        streamer_config: StreamerConfiguration = default_streamer,
        m: int | None = None,
        n: int | None = None,
        k: int | None = None,
        custom_ordering: list[int] | None = None,
    ) -> None:
        super().__init__(streamer_config, custom_ordering)

        if m is not None:
            self.m = m
        if n is not None:
            self.n = n
        if k is not None:
            self.k = k

        # calculate output serializer ratio
        NB_ELEMENTS_PER_PORT = 2
        nb_elements = self.m * self.n
        nb_output_ports = prod(streamer_config.streamers[-1].spatial_dims)
        self.serializer_ratio = nb_elements // (NB_ELEMENTS_PER_PORT * nb_output_ports)

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
            *(f"shift_{i}" for i in range(ceil(self.n / 4))),
            # 8 separate mult values
            *(f"mult_{i}" for i in range(self.n)),
            "temporal_loop_bound",
            "bypassSIMD",
        )

        self.launch_fields = (*self.streamer_launch_fields, "launch_gemmx")

    @classmethod
    def from_config(cls, config: AcceleratorConfig) -> Self:
        assert isinstance(config, GemmxConfig)

        # TODO: get this in the config
        remap: tuple[StreamerOpts, ...]
        if config.m == config.n == config.k:
            transpose = (TransposeExtension(),)
        else:
            transpose = ()
        if config.m > 1:
            remap = (HasAddressRemap(),)
        else:
            remap = ()

        streamer_config = StreamerConfiguration(
            [
                Streamer(  # A
                    StreamerType.Reader,
                    temporal_dims=("n",) * config.streamers[0].temporal_dims,
                    spatial_dims=config.streamers[0].spatial_dims,
                    opts=transpose + remap,
                ),
                Streamer(  # B
                    StreamerType.Reader,
                    temporal_dims=("n",) * config.streamers[1].temporal_dims,
                    spatial_dims=config.streamers[1].spatial_dims,
                    opts=transpose + remap,
                ),
                Streamer(  # D8
                    StreamerType.Writer,
                    temporal_dims=("r",) + ("n",) * (config.streamers[2].temporal_dims - 1),
                    spatial_dims=config.streamers[2].spatial_dims,
                    opts=remap,
                ),
                Streamer(  # C
                    StreamerType.Reader,
                    temporal_dims=("r",) + ("n",) * (config.streamers[3].temporal_dims - 1),
                    spatial_dims=config.streamers[3].spatial_dims,
                    opts=(
                        HasChannelMask(),
                        HasBroadcast(),
                    )
                    + remap,
                ),
                Streamer(  # D32
                    StreamerType.Writer,
                    temporal_dims=("r",) + ("n",) * (config.streamers[4].temporal_dims - 1),
                    spatial_dims=config.streamers[4].spatial_dims,
                    opts=remap,
                ),
            ],
        )

        return cls(streamer_config, config.m, config.n, config.k, config.custom_ordering)

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX GEMMX accelerator op
        """

        # base address:
        base_addr = 0x3C0

        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        nb_shifts = ceil(self.n / 4)
        nb_mults = self.n

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
                **{f"shift_{i}": addr_next + 6 + i for i in range(nb_shifts)},
                **{f"mult_{i}": addr_next + 6 + nb_shifts + i for i in range(nb_mults)},
                "temporal_loop_bound": addr_next + 6 + nb_shifts + nb_mults,
                "bypassSIMD": addr_next + 7 + nb_shifts + nb_mults,
            },
            {**streamer_launch, "launch_gemmx": addr_next + 8 + nb_shifts + nb_mults},
            addr_next + 8 + nb_shifts + nb_mults,
        )
        op.attributes["streamer_config"] = self.streamer_config
        return op

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        if not isinstance(op, snax_stream.StreamingRegionOp):
            return []
        else:
            args, launch_attrs = self._generate_setup_vals(op)

            ops_to_insert: Sequence[Operation] = []
            # insert ops to calculate arguments
            for new_ops, _ in args:
                ops_to_insert.extend(new_ops)

            result = [
                *ops_to_insert,
                setup := accfg.SetupOp([val for _, val in args], self.fields, self.name),
                launch_val := arith.ConstantOp(builtin.IntegerAttr.from_int_and_width(1, 5)),
                token := accfg.LaunchOp([launch_val, launch_val], self.launch_fields, setup),
                accfg.AwaitOp(token),
            ]

            for key, value in launch_attrs.items():
                token.attributes[key] = value

            return result

    def _generate_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> tuple[Sequence[tuple[Sequence[Operation], SSAValue]], dict[str, Attribute]]:
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

        launch_attrs: dict[str, Attribute] = {}

        if isinstance(qmac := generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            if yield_op.arguments[0].type == dart.StreamType(builtin.IntegerType(8)):
                i8_out = True
                last_pattern = op.stride_patterns.data[2]
            else:
                i8_out = False
                last_pattern = op.stride_patterns.data[-1]
            # compute knm: fix n = 1
            n = 1
            # count the number of non-reducing inner bounds (output stride != 0)
            first_non_reducing_bound = next(
                i for i, stride in enumerate(last_pattern.temporal_strides) if stride.data != 0
            )
            m = prod(bound.data for bound in last_pattern.upper_bounds.data[first_non_reducing_bound:]) // n
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
                    rescale_op := region_yield.prev_op.body.block.first_op,
                    kernel.RescaleOp,
                ):
                    max_int_val = rescale_op.max_int.value.data
                    min_int_val = rescale_op.min_int.value.data
                    double_round_val = rescale_op.double_round.value.data
                    shift_vals_int = cast(tuple[int, ...], rescale_op.shift.get_values())
                    if len(shift_vals_int) == 1:
                        shift_vals_int = (shift_vals_int[0],) * self.n
                    mult_vals_int = cast(tuple[int, ...], rescale_op.multiplier.get_values())
                    if len(mult_vals_int) == 1:
                        mult_vals_int = (mult_vals_int[0],) * self.n
                    zp_in_val = rescale_op.input_zp.value.data
                    zp_out_val = rescale_op.output_zp.value.data
                else:
                    max_int_val = 127
                    min_int_val = -128
                    double_round_val = 0
                    shift_vals_int = (9,) * self.n
                    mult_vals_int = (1,) * self.n
                    zp_in_val = 0
                    zp_out_val = 0

                m_val = arith.ConstantOp.from_int_and_width(m, 32)
                ops_to_add.append(m_val)
                loop_bound = m_val

                max_int = arith.ConstantOp.from_int_and_width(max_int_val, i32)
                min_int = arith.ConstantOp.from_int_and_width(min_int_val, i32)
                double_round = arith.ConstantOp.from_int_and_width(double_round_val, i32)
                shifts = [arith.ConstantOp.from_int_and_width(shift_val, i32) for shift_val in shift_vals_int]
                mults = [arith.ConstantOp.from_int_and_width(mult_val, i32) for mult_val in mult_vals_int]
                zp_in = arith.ConstantOp.from_int_and_width(zp_in_val, i32)
                zp_out = arith.ConstantOp.from_int_and_width(zp_out_val, i32)
                ops_to_add.extend([max_int, min_int, double_round, *shifts, *mults, zp_in, zp_out])

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

                shift_vals: Sequence[SSAValue] = []
                for i in range(0, len(shifts), 4):  # 4 8-bit shift vals per 32-bit csr
                    shift_bitlist = list(pack_bitlist(shifts[i : i + 4][::-1], (24, 16, 8, 0)))
                    ops_to_add.extend(shift_bitlist)
                    shift_vals.append(shift_bitlist[-1].results[0])

                if len(shift_vals) > ceil(self.n / 4):
                    launch_attrs["shift_vals"] = cast(DenseArrayBase, rescale_op.shift)  # pyright: ignore
                    shift_vals = shift_vals[: ceil(self.n / 4)]

                mult_vals = [mult.result for mult in mults]
                if len(mult_vals) > self.n:
                    launch_attrs["mult_vals"] = cast(DenseArrayBase, rescale_op.multiplier)  # pyright: ignore
                    launch_attrs["m"] = m_val.value
                    mult_vals = mult_vals[: self.n]

            else:
                loop_bound = c0
                csr0 = c0.result
                csr1 = c0.result
                shift_vals = [c0.result for _ in range(ceil(self.n / 4))]
                mult_vals = [c1.result for _ in range(self.n)]

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
            shift_val = cast(int, rescale.shift.get_values()[0])
            mult_val = cast(int, rescale.multiplier.get_values()[0])
            shift = arith.ConstantOp.from_int_and_width(shift_val, i32)
            mult = arith.ConstantOp.from_int_and_width(mult_val, i32)
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

            shift_vals = [shift_bitlist[-1].results[0] for _ in range(ceil(self.n / 4))]
            mult_vals = (mult.result for _ in range(ceil(self.n / 4)))

            loop_bound = prod(x.data for x in op.stride_patterns.data[0].upper_bounds)
            loop_bound = arith.ConstantOp.from_int_and_width(loop_bound, i32)
            ops_to_add.append(loop_bound)

        else:
            raise NotImplementedError()

        knm: list[tuple[tuple[Operation], OpResult]] = [
            (((cst := arith.ConstantOp.from_int_and_width(val, 32)),), cst.result) for val in (k, n, m)
        ]

        return (
            [
                *streamer_setup_vals,
                *knm,
                ([c0, c1, *ops_to_add], subtractions),  # subtractions
                ([], csr0),  # csr0
                ([], csr1),  # csr1
                *(([], x) for x in shift_vals),
                *(([], x) for x in mult_vals),
                ([], loop_bound.result),  # temporal_loop_bound
                ([], bypassSIMD),  # bypassSIMD
            ],
            launch_attrs,
        )

    def lower_acc_launch(self, launch_op: accfg.LaunchOp, acc_op: accfg.AcceleratorOp) -> Sequence[Operation]:
        if "mult_vals" not in launch_op.attributes:
            return super().lower_acc_launch(launch_op, acc_op)

        # special case for channel-specific quantization scales:
        # for this to work, the schedule must be output-channel stationary
        # at this point, we just assume that this is the case.

        def csr_op(addr: SSAValue, value: SSAValue) -> Operation:
            return llvm.InlineAsmOp(
                "csrw $0, $1",
                # I = any 12 bit immediate, K = any 5 bit immediate
                # The K allows LLVM to emit an `csrrwi` instruction,
                # which has room for one 5 bit immediate only.
                "I, rK",
                [addr, value],
                has_side_effects=True,
            )

        field_to_csr_launch = dict(acc_op.launch_field_items())
        field_to_csr = dict(acc_op.field_items())
        ops: Sequence[Operation] = []

        launch_values = {field: val for field, val in launch_op.iter_params()}

        # add address vals:
        ops.append(addr_gemmx := arith.ConstantOp(field_to_csr_launch["launch_gemmx"]))
        ops.append(addr_streamer := arith.ConstantOp(field_to_csr_launch["launch_streamer"]))

        # overwrite m and temporal loop bound:
        m = launch_op.attributes["m"]
        assert isa(m, builtin.IntegerAttr[IntegerType])
        mult_vals = launch_op.attributes["mult_vals"]
        assert isinstance(mult_vals, DenseArrayBase)
        shift_vals = launch_op.attributes["shift_vals"]
        assert isinstance(shift_vals, DenseArrayBase)
        new_m = builtin.IntegerAttr.from_int_and_width(m.value.data // (len(mult_vals) // self.n), m.type.width.data)

        ops.append(new_m_val := arith.ConstantOp(new_m))
        ops.append(m_addr := arith.ConstantOp(field_to_csr["M"]))
        ops.append(loop_bound_addr := arith.ConstantOp(field_to_csr["temporal_loop_bound"]))
        ops.append(csr_op(m_addr.result, new_m_val.result))
        ops.append(csr_op(loop_bound_addr.result, new_m_val.result))

        # launch streamer once
        ops.append(csr_op(addr_streamer.result, launch_values["launch_streamer"]))

        shift_vals = cast(
            tuple[int, ...],
            cast(DenseArrayBase, launch_op.attributes["shift_vals"]).get_values(),
        )
        mult_vals = cast(
            tuple[int, ...],
            cast(DenseArrayBase, launch_op.attributes["mult_vals"]).get_values(),
        )

        for i in range(len(mult_vals) // self.n):
            # reprogram the shift and mult values
            shifts = shift_vals[i * self.n : i * self.n + self.n]
            for j in range(0, len(shifts), 4):  # 4 8-bit shift vals per 32-bit csr
                shift_bitlist = list(pack_bitlist(shifts[j : j + 4][::-1], (24, 16, 8, 0)))
                ops.extend(shift_bitlist)
                ops.append(csr_addr := arith.ConstantOp(field_to_csr[f"shift_{j // 4}"]))
                ops.append(csr_op(csr_addr.result, shift_bitlist[-1].results[0]))

            mults = mult_vals[i * self.n : i * self.n + self.n]
            for j in range(len(mults)):
                ops.append(mult_val := arith.ConstantOp.from_int_and_width(mults[j], i32))
                ops.append(csr_addr := arith.ConstantOp(field_to_csr[f"mult_{j}"]))
                ops.append(csr_op(csr_addr.result, mult_val.result))

            # launch the accelerator for each output channel // 8
            ops.append(csr_op(addr_gemmx.result, launch_values["launch_gemmx"]))

            # await the accelerator only
            ops.append(cst_0 := arith.ConstantOp.from_int_and_width(0, 32))
            ops.append(csr_op(addr_gemmx.result, cst_0.result))

        return ops

    def get_template(self, op: dart.StreamingRegionOpBase) -> Template:
        if self.m == 1:
            return self.get_matvec_template(op)
        return self.get_matmul_template(op)

    def get_matmul_template(self, op: dart.StreamingRegionOpBase) -> Template:
        assert isinstance(generic_op := op.body.block.first_op, dart.GenericOp)
        if isinstance(generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            # matmul
            m, n, k = (AffineDimExpr(i) for i in range(3))
            template = [
                AffineMap(3, 0, (m, k)),
                AffineMap(3, 0, (k, n)),
                AffineMap(3, 0, (m, n)),
            ]
            template_bounds = (self.m, self.n, self.k)

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
            template_bounds = (self.m, self.k)

        if not isinstance(generic_op.next_op, dart.YieldOp):
            raise RuntimeError("unsupported kernel")

        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_matvec_template(self, op: dart.StreamingRegionOpBase) -> Template:
        assert isinstance(generic_op := op.body.block.first_op, dart.GenericOp)
        if isinstance(generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            # matmul
            n, k = (AffineDimExpr(i) for i in range(2))
            template = [
                AffineMap(2, 0, (k,)),
                AffineMap(2, 0, (k, n)),
                AffineMap(2, 0, (n,)),
            ]
            template_bounds = (self.n, self.k)

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
            raise NotImplementedError()

        if not isinstance(generic_op.next_op, dart.YieldOp):
            raise RuntimeError("unsupported kernel")

        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, op: dart.StreamingRegionOpBase) -> Sequence[Streamer]:
        if len(op.patterns) == 3:
            # matmul, no add
            if op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(32)):
                streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 4)]
            elif op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(8)):
                streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 2)]
            else:
                raise NotImplementedError("Unsupported type for snax_gemmx accelerator")
        elif len(op.patterns) == 4:
            # gemm with add
            if op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(32)):
                streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 3, 4)]
            elif op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(8)):
                streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 3, 2)]
            else:
                raise NotImplementedError("Unsupported type for snax_gemmx accelerator")
        else:
            streamers = [self.streamer_config.data.streamers[i] for i in (3, 2)]
        return streamers

    def set_stride_patterns(
        self,
        op: dart.AccessPatternOp,
        snax_stride_patterns: Sequence[snax_stream.StridePattern],
    ) -> tuple[
        Sequence[SSAValue],
        Sequence[SSAValue],
        Sequence[snax_stream.StridePattern],
        Sequence[Operation],
    ]:
        snax_stride_patterns = list(snax_stride_patterns)
        new_inputs: list[SSAValue[Attribute]] = list(op.inputs)
        new_outputs: list[SSAValue] = list(op.outputs)
        ops_to_add: list[Operation] = []

        empty_pattern = snax_stream.StridePattern(upper_bounds=[0] * 3, temporal_strides=[0] * 3, spatial_strides=[0])
        if len(snax_stride_patterns) == 3:
            if op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(32)):
                # matmul, int32 output
                # insert empty patterns for D8 and zero pattern for C
                snax_stride_patterns.insert(2, empty_pattern)
                new_inputs.append(op.outputs[0])

                # insert same pattern for C as for D32
                snax_stride_patterns.insert(
                    3,
                    snax_stream.StridePattern(
                        upper_bounds=snax_stride_patterns[3].upper_bounds,
                        temporal_strides=snax_stride_patterns[3].temporal_strides,
                        spatial_strides=snax_stride_patterns[3].spatial_strides,
                    ),
                )

                # point C to 0
                new_inputs.append(op.outputs[0])
                # ops_to_add.append(
                #     # zero pointer will generate 0 values
                #     ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                # )
                # new_inputs.append(ptr.result)

            elif op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(8)):
                new_inputs.append(new_outputs.pop())
                # matmul, int8 output
                # for C32:
                # TODO: 8 here still refers to hardcoded TCDM bank width
                nb_output_spats = len(self.streamer_config.data.streamers[-1].spatial_dims)
                snax_stride_patterns.append(
                    snax_stream.StridePattern(
                        upper_bounds=[x.data for x in snax_stride_patterns[2].upper_bounds.data[:-1]]
                        + [snax_stride_patterns[2].upper_bounds.data[-1].data * self.serializer_ratio],
                        temporal_strides=[x.data for x in snax_stride_patterns[2].temporal_strides],
                        spatial_strides=[8 * self.streamer_config.data.streamers[2].spatial_dims[-1], 8][
                            -nb_output_spats:
                        ],
                    )
                )
                ops_to_add.append(
                    # zero pointer will generate 0 values
                    ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
                )
                new_inputs.append(ptr.result)
                # for D32
                snax_stride_patterns.append(
                    snax_stream.StridePattern(
                        upper_bounds=[0, 0, 0],
                        temporal_strides=[0, 0, 0],
                        spatial_strides=[0] * nb_output_spats,
                    )
                )
                new_inputs.append(op.outputs[0])

        elif len(snax_stride_patterns) == 4:
            # gemm
            #
            # for a gemm, the 8bit-output port D8 are unused, so we create
            # empty patterns for them here
            if op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(32)):
                snax_stride_patterns.insert(2, empty_pattern)
                new_inputs.insert(2, op.inputs[-1])
            elif op.body.block.arg_types[-1] == dart.StreamType(builtin.IntegerType(8)):
                d8_pattern = snax_stride_patterns.pop()
                d8_input = new_outputs.pop()
                snax_stride_patterns.insert(2, d8_pattern)
                new_inputs.insert(2, d8_input)

                # for D32:
                snax_stride_patterns.append(
                    snax_stream.StridePattern(
                        upper_bounds=[0, 0, 0],
                        temporal_strides=[0, 0, 0],
                        spatial_strides=[0, 0],
                    )
                )
                new_inputs.append(op.inputs[-1])
            else:
                raise RuntimeError("unsupported case")

        else:
            # simd
            # to calculate only simd, we calculate the result
            # of D8 = rescale(AxB + C)
            # create zero patterns for A and B such that D8 = rescale(C)
            # create empty pattern for D32
            # do not use new outputs
            new_inputs.append(new_outputs.pop())

            zero_pattern = snax_stream.StridePattern(
                upper_bounds=snax_stride_patterns[0].upper_bounds,
                temporal_strides=[0] * len(snax_stride_patterns[0].upper_bounds),
                spatial_strides=[8],
            )

            # read zeros from tcdm (must make sure there are zeros at these addresses)
            # in the new streamer this can be fixed with byte masking
            snax_stride_patterns.insert(0, zero_pattern)
            ops_to_add.append(ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
            new_inputs.insert(0, ptr.result)
            snax_stride_patterns.insert(1, zero_pattern)
            ops_to_add.append(ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType()))
            new_inputs.insert(1, ptr.result)

            # flip D8 and C such that they are in the right order
            snax_stride_patterns.append(snax_stride_patterns.pop(2))
            new_inputs.append(new_inputs.pop(2))

            # empty pattern for D32
            snax_stride_patterns.append(empty_pattern)
            # dummy base pointer for D32
            new_inputs.append(op.inputs[-1])

            # make last spatial stride patterns 2d
            # the spatial strides do not matter here (i think)
            snax_stride_patterns[-2] = snax_stream.StridePattern(
                upper_bounds=snax_stride_patterns[-2].upper_bounds,
                temporal_strides=snax_stride_patterns[-2].temporal_strides,
                spatial_strides=[8, 64],
            )
            snax_stride_patterns[-1] = snax_stream.StridePattern(
                upper_bounds=snax_stride_patterns[-1].upper_bounds,
                temporal_strides=snax_stride_patterns[-1].temporal_strides,
                spatial_strides=[8, 64],
            )

        return new_inputs, new_outputs, snax_stride_patterns, ops_to_add
