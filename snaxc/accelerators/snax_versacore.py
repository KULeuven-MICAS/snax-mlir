from collections.abc import Sequence
from math import prod
from typing import Self

from xdsl.dialects import arith, builtin
from xdsl.dialects.builtin import i8, i32
from xdsl.ir import Attribute, BlockArgument, Operation, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap

from snaxc.accelerators.configurable_accelerator import ConfigurableAccelerator
from snaxc.accelerators.dispatching import DispatchTemplate, SupportedKernel
from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier3,
    SNAXStreamer,
)
from snaxc.accelerators.streamers import (
    HasAddressRemap,
    HasBroadcast,
    HasChannelMask,
)
from snaxc.accelerators.streamers.streamers import (
    Streamer,
    StreamerConfiguration,
    StreamerType,
)
from snaxc.dialects import accfg, dart, kernel, snax_stream
from snaxc.ir.dart.access_pattern import Template, TemplatePattern
from snaxc.tools.configs import AcceleratorConfig, VersaCoreConfig, VersaCoreMapping
from snaxc.util.pack_bitlist import pack_bitlist

default_streamer = StreamerConfiguration(
    [
        Streamer(  # A
            StreamerType.Reader,
            temporal_dims=("n", "n", "n", "n", "n", "n"),
            spatial_dims=(32,),
            opts=(HasChannelMask(), HasAddressRemap()),
        ),
        Streamer(  # B
            StreamerType.Reader,
            temporal_dims=("n", "n", "n"),
            spatial_dims=(32,),
            opts=(HasChannelMask(), HasAddressRemap()),
        ),
        Streamer(  # C
            StreamerType.Reader,
            temporal_dims=("r", "n", "n", "n"),
            spatial_dims=(16,),
            opts=(HasChannelMask(), HasBroadcast()),
        ),
        Streamer(  # D32
            StreamerType.Writer,
            temporal_dims=("r", "n", "n", "n"),
            spatial_dims=(16,),
            opts=(HasChannelMask(), HasAddressRemap()),
        ),
    ],
)


class SNAXVersaCoreAccelerator(
    SNAXAccelerator, SNAXStreamer, DispatchTemplate, SNAXPollingBarrier3, ConfigurableAccelerator
):
    """
    Accelerator Interface class for SNAX Versacore accelerator
    """

    name = "snax_versacore"

    configs: list[VersaCoreMapping] = [VersaCoreMapping(m=16, n=16, k=16)]

    # serializer_ratio: int

    supported_kernels = (SupportedKernel(kernel.QMacOp, (i8, i8, i32, i32, i32)),)

    def __init__(
        self,
        streamer_config: StreamerConfiguration = default_streamer,
        versacore_configs: Sequence[VersaCoreMapping] | None = None,
    ) -> None:
        super().__init__(streamer_config)

        # calculate output serializer ratio
        # NB_ELEMENTS_PER_PORT = 2
        # nb_elements = self.m * self.n
        # nb_output_ports = prod(streamer_config.streamers[-1].spatial_dims)
        # self.serializer_ratio = nb_elements // (NB_ELEMENTS_PER_PORT * nb_output_ports)

        self.fields = (
            *self.streamer_setup_fields,
            "overwrite_accum",
            "accum_bound",
            "output_bound",
            "subtractions",
            "array_shape_cfg",
            "data_type_cfg",
        )

        self.launch_fields = (*self.streamer_launch_fields, "launch_versacore")

    @classmethod
    def from_config(cls, config: AcceleratorConfig) -> Self:
        assert isinstance(config, VersaCoreConfig)
        streamer_config = StreamerConfiguration(
            [
                Streamer(  # A
                    StreamerType.Reader,
                    temporal_dims=("n",) * config.streamers[0].temporal_dims,
                    spatial_dims=tuple(config.streamers[0].spatial_dims),
                    opts=(HasChannelMask(), HasAddressRemap()),
                ),
                Streamer(  # B
                    StreamerType.Reader,
                    temporal_dims=("n",) * config.streamers[1].temporal_dims,
                    spatial_dims=tuple(config.streamers[1].spatial_dims),
                    opts=(HasChannelMask(), HasAddressRemap()),
                ),
                Streamer(  # C
                    StreamerType.Reader,
                    temporal_dims=("r", "r") + ("n",) * (config.streamers[2].temporal_dims - 2),
                    spatial_dims=config.streamers[2].spatial_dims,
                    opts=(HasChannelMask(), HasBroadcast()),
                ),
                Streamer(  # D32
                    StreamerType.Writer,
                    temporal_dims=("r", "r") + ("n",) * (config.streamers[3].temporal_dims - 2),
                    spatial_dims=config.streamers[3].spatial_dims,
                    opts=(HasChannelMask(), HasAddressRemap()),
                ),
            ],
        )

        return cls(streamer_config, config.configs)

    def generate_acc_op(self) -> accfg.AcceleratorOp:
        """
        Return a SNAX Versacore accelerator op
        """

        # base address:
        base_addr = 0x3C0

        addr_next, streamer_setup = self.get_streamer_setup_dict(base_addr)
        addr_next, streamer_launch = self.get_streamer_launch_dict(addr_next)

        op = accfg.AcceleratorOp(
            self.name,
            {
                **streamer_setup,
                "overwrite_accum": addr_next + 0,
                "accum_bound": addr_next + 1,
                "output_bound": addr_next + 2,
                "subtractions": addr_next + 3,
                "array_shape_cfg": addr_next + 4,
                "data_type_cfg": addr_next + 5,
            },
            {**streamer_launch, "launch_versacore": addr_next + 6},
            addr_next + 7,
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
        assert isinstance(op.body.block.last_op, dart.YieldOp)

        launch_attrs: dict[str, Attribute] = {}

        if isinstance(qmac := generic_op.body.block.first_op, kernel.QMacOp):
            last_pattern = op.stride_patterns.data[-1]
            # count the number of non-reducing bounds (output stride != 0)
            output_bound = prod(
                bound.data
                for bound, stride in zip(
                    last_pattern.upper_bounds,
                    last_pattern.temporal_strides,
                )
                if stride.data != 0
            )
            output_bound_op = arith.ConstantOp.from_int_and_width(output_bound, 32)
            accum_bound = prod(x.data for x in op.stride_patterns.data[0].upper_bounds) // output_bound
            accum_bound_op = arith.ConstantOp.from_int_and_width(accum_bound, 32)

            assert isinstance(qmac.zp_lhs, BlockArgument)
            zp_a = generic_op.inputs[qmac.zp_lhs.index]
            assert isinstance(qmac.zp_rhs, BlockArgument)
            zp_b = generic_op.inputs[qmac.zp_rhs.index]

            # bitwise and with 8b'11111111 to avoid the sign bits extending the 8-bit field
            # when bitlist packing
            ops_to_add.append(cst255 := arith.ConstantOp.from_int_and_width(255, 32))
            ops_to_add.append(zp_a := arith.AndIOp(zp_a, cst255))
            ops_to_add.append(zp_b := arith.AndIOp(zp_b, cst255))

            bitlist = list(pack_bitlist((zp_a, zp_b), [0, 8]))
            ops_to_add.extend(bitlist)
            subtractions = bitlist[-1].results[0]
        else:
            raise NotImplementedError()

        return (
            [
                *streamer_setup_vals,
                # overwrite accum
                ([c0, c1], c1.result),
                # accum bound
                ([accum_bound_op], accum_bound_op.result),
                # output bound
                ([output_bound_op], output_bound_op.result),  # subtractions
                # subtractions
                (ops_to_add, subtractions),  # temporal_loop_bound
                # array configuration
                ([], c0.result),
                # data type configuration
                ([], c0.result),
            ],
            launch_attrs,
        )

    def get_template(self, op: dart.StreamingRegionOpBase) -> Template:
        assert isinstance(generic_op := op.body.block.first_op, dart.GenericOp)
        if isinstance(generic_op.body.block.first_op, kernel.QMacOp | kernel.MacOp):
            # matmul
            m, n, k = (AffineDimExpr(i) for i in range(3))
            template = [
                AffineMap(3, 0, (m, k)),
                AffineMap(3, 0, (k, n)),
                AffineMap(3, 0, (m, n)),
            ]
            template_bounds = (self.configs[0].m, self.configs[0].n, self.configs[0].k)

            if isinstance(generic_op.next_op, dart.GenericOp):
                generic_op = generic_op.next_op
                if isinstance(generic_op.body.block.first_op, kernel.AddOp):
                    # gemm, add c pattern that is equal to output pattern
                    template += [template[-1]]
                else:
                    raise RuntimeError("unsupported kernel")
        else:
            raise NotImplementedError("unsupported kernel")

        if not isinstance(generic_op.next_op, dart.YieldOp):
            raise RuntimeError("unsupported kernel")

        return Template(TemplatePattern(template_bounds, tp) for tp in template)

    def get_streamers(self, op: dart.StreamingRegionOpBase) -> Sequence[Streamer]:
        if len(op.patterns) == 3:
            # matmul, no add
            streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 3)]
        elif len(op.patterns) == 4:
            streamers = [self.streamer_config.data.streamers[i] for i in (0, 1, 2, 3)]
        else:
            raise Exception
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

        if len(snax_stride_patterns) == 3:
            # insert same pattern for C as for D32
            snax_stride_patterns.insert(
                2,
                snax_stream.StridePattern(
                    upper_bounds=snax_stride_patterns[2].upper_bounds,
                    temporal_strides=snax_stride_patterns[2].temporal_strides,
                    spatial_strides=snax_stride_patterns[2].spatial_strides,
                ),
            )

            # point C to 0
            ops_to_add.append(
                # zero pointer will generate 0 values
                ptr := arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
            )
            new_inputs.append(ptr.result)

        elif len(snax_stride_patterns) == 4:
            # nothing special to do
            pass

        return new_inputs, new_outputs, snax_stride_patterns, ops_to_add
