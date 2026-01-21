from collections.abc import Sequence

from xdsl.dialects import arith, builtin, linalg
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter

from snaxc.accelerators.snax import (
    SNAXAccelerator,
    SNAXPollingBarrier3,
    SNAXStreamer,
)
from snaxc.dialects import accfg, dart, phs, snax_stream
from snaxc.ir.dart.access_pattern import Template
from snaxc.phs.decode import decode_abstract_graph
from snaxc.phs.encode import convert_generic_body_to_phs
from snaxc.phs.template_spec import TemplateSpec


class SNAXPHSAccelerator(SNAXAccelerator, SNAXPollingBarrier3, SNAXStreamer):
    """
    Accelerator interface class for the SNAX PHS accelerator.
    """

    def __init__(self, pe: phs.PEOp, template_spec: TemplateSpec) -> None:
        self.pe = pe

        acc_name = pe.properties["sym_name"]
        assert isinstance(acc_name, StringAttr)
        self.name = acc_name.data

        self.template_spec = template_spec
        super().__init__(template_spec.get_streamer_config())
        self.fields = (*self.streamer_setup_fields, "alu_mode", "loop_bound_alu")
        self.launch_fields = (*self.streamer_launch_fields, "launch_alu")

    def convert_to_acc_ops(self, op: Operation) -> Sequence[Operation]:
        """
        Lowers the operation to a sequence of acc_ops.
        """

        # linalg.generic lowering is stil hardcoded, but kept until
        # lowering from linalg -> snax_stream is complete
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

    def get_switch_values(
        self, op: linalg.GenericOp | dart.GenericOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        self.pe: phs.PEOp
        candidate_pe = convert_generic_body_to_phs(op, self.name, PatternRewriter(op))
        switch_values = decode_abstract_graph(self.pe, candidate_pe)
        ops = [arith.ConstantOp.from_int_and_width(value, 32) for value in switch_values]
        return [([op], op.results[0]) for op in ops]

    def _generate_stream_setup_vals(
        self, op: snax_stream.StreamingRegionOp
    ) -> Sequence[tuple[Sequence[Operation], SSAValue]]:
        loop_bound = arith.ConstantOp.from_int_and_width(op.stride_patterns.data[0].upper_bounds.data[0], 32)
        generic = op.regions[0].ops.first
        assert isinstance(generic, linalg.GenericOp) or isinstance(generic, dart.GenericOp)

        return [
            *self._generate_streamer_setup_vals(op),
            *self.get_switch_values(generic),
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

    def get_template(self, op: dart.StreamingRegionOpBase) -> Template:
        return self.template_spec.get_dart_template()
