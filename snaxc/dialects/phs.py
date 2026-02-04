from __future__ import annotations

from collections.abc import Iterator, Sequence

from xdsl.dialects.builtin import (
    I64,
    ArrayAttr,
    DictionaryAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    StringAttr,
)
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Attribute, Block, BlockArgument, Dialect, Region, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    Operation,
    irdl_op_definition,
    lazy_traits_def,
    operand_def,
    opt_prop_def,
    prop_def,
    region_def,
    result_def,
    traits_def,
    var_operand_def,
    var_region_def,
    var_result_def,
)
from xdsl.parser import Parser, SymbolRefAttr
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator, Pure, SymbolOpInterface, SymbolTable
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    """
    Terminator operation for phs operations
    """

    name = "phs.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasParent(PEOp, ChooseOp),
            Pure(),
        )
    )

    @property
    def data_operands(self) -> Sequence[SSAValue]:
        """
        Return every operand except for switches.
        Since there are no switches here, it should return all operands
        """
        return self._operands


@irdl_op_definition
class PEOp(IRDLOperation):
    """
    Processing Element operation - an abstract representation of a single
    processing element that can possibly support multiple sequences of operations
    with the use of different switches.
    """

    name = "phs.pe"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    function_type = prop_def(FunctionType)

    body = region_def("single_block")

    function_type = prop_def(FunctionType)

    switch_no = prop_def(IntegerAttr[I64])  # the last switch_no block_args are considered switches

    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface(), SymbolTable())

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
        switch_no: int = 0,
        region: Region | None = None,
        *,
        arg_attrs: ArrayAttr[DictionaryAttr] | None = None,
        res_attrs: ArrayAttr[DictionaryAttr] | None = None,
    ):
        if isinstance(function_type, tuple):
            inputs, outputs = function_type
            function_type = FunctionType.from_lists(inputs, outputs)
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        super().__init__(
            properties={
                "sym_name": StringAttr(name),
                "function_type": function_type,
                "switch_no": IntegerAttr.from_int_and_width(switch_no, 64),
                "arg_attrs": arg_attrs,
                "res_attrs": res_attrs,
            },
            regions=[region],
        )

    def verify_(self) -> None:
        # If this is an empty region
        if len(self.body.blocks) == 0:
            return

        block = self.body.block
        block_arg_types = block.arg_types
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException("Expected block arguments to have the same types as the function input types")

    @staticmethod
    def from_operations(acc_ref: SymbolRefAttr, operations: Sequence[Operation]) -> PEOp:
        """
        Utility constructor that fills up a PE operation with a single ChooseOp that supports
        all operations given in "operations"
        """
        # There will be only on switch of index type
        switch_types = [IndexType()]

        # Check whether all operations have equal types
        in_types = operations[0].operand_types
        out_types = operations[0].result_types
        for operation in operations[1:]:
            for typ_a, typ_b in zip(operations[0].operand_types, operation.operand_types, strict=True):
                assert type(typ_a) is type(typ_b), "expect operand types of all operations to be equal"
            for typ_a, typ_b in zip(operations[0].result_types, operation.result_types, strict=True):
                assert type(typ_a) is type(typ_b), "expect result types of all operations to be equal"

        # Construct a new block based on the input of the
        block_inputs = [*in_types, *switch_types]
        block = Block(arg_types=block_inputs)

        # Map block args to inputs and outputs to yield
        data_operands = block.args[:-1]
        switch = block.args[-1]

        # Create the new operation
        choose_op = ChooseOp.from_operations("0", data_operands, switch, operations=operations, result_types=out_types)
        block.add_ops(
            [
                choose_op,
                YieldOp(choose_op),
            ]
        )
        pe_op = PEOp(acc_ref.string_value(), (block_inputs, out_types), 1, Region(block))
        return pe_op

    def get_choose_op(self, symbol_name: str) -> ChooseOp | None:
        """
        Get a specific choose op inside the PEOp by symbol name
        """
        t = self.get_trait(SymbolTable)
        assert t is not None, "No SymbolTable present in current operation"
        choose_op = t.lookup_symbol(self, symbol_name)
        if choose_op is None:
            return choose_op
        else:
            assert isinstance(choose_op, ChooseOp)
            return choose_op

    def get_terminator(self) -> YieldOp:
        """
        Get the terminating operation inside the PEOp body.
        """
        yield_op = self.body.ops.last
        assert isinstance(yield_op, YieldOp)
        return yield_op

    def add_switch(self) -> BlockArgument:
        """
        Add an extra switch to the PE operation
        """
        block = self.body.block
        # Add new switch at the end
        self.switch_no = IntegerAttr.from_int_and_width(self.switch_no.value.data + 1, 64)
        self.function_type = FunctionType.from_lists(
            list(self.function_type.inputs) + [IndexType()], list(self.function_type.outputs)
        )
        return block.insert_arg(IndexType(), len(block.args))

    def remove_switch(self, switch: BlockArgument) -> None:
        assert self.switch_no.value.data != 0, "No switches to remove!"
        self.body.block.erase_arg(switch)
        self.switch_no = IntegerAttr.from_int_and_width(self.switch_no.value.data - 1, 64)
        input_types = list(self.function_type.inputs)
        input_types.pop(switch.index)
        self.function_type = FunctionType.from_lists(input_types, list(self.function_type.outputs))

    def _get_block_args(self) -> list[BlockArgument[Attribute]]:
        block = self.body.block
        return list(block.args)

    def get_switches(self) -> list[BlockArgument[Attribute]]:
        """
        Get BlockArguments that relate to switch input in PE operation
        """
        num_switches = self.switch_no.value.data
        if num_switches == 0:
            return []
        # The last switch_no arguments are the switches
        return self._get_block_args()[-num_switches:]

    def get_true_switches(self) -> int:
        """
        Get amount of switches that actually get called in PE operation
        """
        count: int = 0
        for switch in self.get_switches():
            switchee = switch.get_user_of_unique_use()
            assert switchee is not None, (
                f"Switch does not drive one choice in the PE (got {switch.uses.get_length()} uses)"
            )
            if isinstance(switchee, ChooseOp):
                if len(list(switchee.operations())) > 1:
                    count += 1
            elif isinstance(switchee, MuxOp):
                count += 1
        return count

    def data_operands(self) -> list[BlockArgument[Attribute]]:
        """
        Get BlockArguments that relate to data input in PE operation
        """
        num_switches = self.switch_no.value.data
        args = self._get_block_args()
        if num_switches == 0:
            return args
        return args[:-num_switches]

    def is_concrete(self) -> bool:
        """
        Verify that a PEOp is fully specified by its body.
        I.e. there's really only one set of operations that can be performed.
        This means that this check will verify whether:
        * All choose_ops in this PEOp have exactly one operation.
        * There are no MuxOps in this PEOp
        """
        for operation in self.body.ops:
            if isinstance(operation, MuxOp):
                # If there's a MuxOp, multiple choices exist and the PE is not concrete
                return False
            elif isinstance(operation, ChooseOp):
                # If there are multiple operations to be chosen, the PE is not concrete
                if len(list(operation.operations())) == 1:
                    continue
                else:
                    return False
            elif isinstance(operation, YieldOp):
                # If we reach the YieldOp without any issue, the PE is concrete
                return True
            else:
                raise NotImplementedError("Don't expect any other operations besides MuxOp, YieldOp or ChooseOp")
        # No operations exist in the body
        raise NotImplementedError("No operations found in the PEOp body")

    def print(self, printer: Printer):
        printer.print_string(" @")
        printer.print_string(self.name_prop.data)
        if self.switch_no.value.data != 0:
            printer.print_string(" with ")
        first_switch = True
        # First print switches
        for block_arg in self.get_switches():
            if not first_switch:
                printer.print_string(", ")
            printer.print_operand(block_arg)
            first_switch = False

        # After switches, print operands
        printer.print_string(" (")
        data_operands = self.data_operands()
        for i, opnd in enumerate(data_operands):
            printer.print_operand(opnd)
            printer.print_string(" : ")
            printer.print_attribute(opnd.type)
            if i != len(data_operands) - 1:
                printer.print_string(", ")

        printer.print_string(") {")
        with printer.indented():
            for op in self.regions[0].block.ops:
                printer.print_string("\n")
                printer.print_op(op)

        printer.print_string("\n}")

    @classmethod
    def parse(cls: type[PEOp], parser: Parser) -> PEOp:
        name_prop = parser.parse_symbol_name()
        with_keyword = parser.parse_optional_keyword("with")

        switches: list[Parser.Argument] = []
        if with_keyword:
            while True:
                arg = parser.parse_optional_argument(expect_type=False)
                if arg is None:
                    break
                arg = arg.resolve(IndexType())
                parser.parse_optional_punctuation(",")
                switches.append(arg)

        data_operands: list[Parser.Argument] = []
        parser.parse_punctuation("(")
        while True:
            arg = parser.parse_optional_argument(expect_type=True)
            if arg is None:
                break
            parser.parse_optional_punctuation(",")
            data_operands.append(arg)
        parser.parse_punctuation(")")

        input_args = [*data_operands, *switches]
        region = parser.parse_region(arguments=input_args)
        yield_op = region.block.ops.last
        assert isinstance(yield_op, YieldOp)
        in_types = [arg.type for arg in input_args]
        out_types = yield_op.operand_types
        pe_op = cls(
            name_prop.data,
            function_type=FunctionType.from_lists(in_types, out_types),
            switch_no=len(switches),
            region=region,
        )
        return pe_op


@irdl_op_definition
class ChooseOp(IRDLOperation):
    """
    Operation to choose between operations contained in its region.
    Very similar to scf.index_switch.
    """

    name = "phs.choose"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    data_operands = var_operand_def()

    switch = operand_def(IndexType)

    # This is quite similar to a scf.index_switch
    default_region = region_def("single_block")
    case_regions = var_region_def("single_block")

    res = var_result_def()

    traits = traits_def(SymbolOpInterface(), IsolatedFromAbove())

    def __init__(
        self,
        name: str,
        data_operands: Sequence[Operation | SSAValue],
        switch: Operation | SSAValue,
        default_region: Region,
        case_regions: Sequence[Region] = [],
        result_types: Sequence[Attribute] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            properties={
                "sym_name": StringAttr(name),
            },
            operands=(data_operands, switch),
            regions=(default_region, case_regions),
            attributes=attr_dict,
            result_types=(result_types,),
        )

    @staticmethod
    def _check_operand_types(
        data_operands: Sequence[Operation | SSAValue], operations: Sequence[Operation]
    ) -> Sequence[Attribute]:
        data_operand_types = [SSAValue.get(data_operand).type for data_operand in data_operands]
        for operation in operations:
            for opnd_type, data_opnd_type in zip(operation.operand_types, data_operand_types, strict=True):
                assert type(opnd_type) is type(data_opnd_type), (
                    f"Operation {operation.name} and phs.choose_op type mismatch ({opnd_type} vs. {data_opnd_type})"
                )
        return data_operand_types

    @staticmethod
    def from_operations(
        name: str,
        data_operands: Sequence[Operation | SSAValue],
        switch: Operation | SSAValue,
        operations: Sequence[Operation],
        result_types: Sequence[Attribute] = [],
    ) -> ChooseOp:
        """
        Utility constructor to construct a ChooseOp from a set of given operations
        """
        # Make sure that all operand types are equal for all operations
        data_operand_types = ChooseOp._check_operand_types(data_operands, operations)
        # Default operation
        default_block = Block(arg_types=data_operand_types)
        value_mapper = {
            SSAValue.get(arg): SSAValue.get(val)
            for arg, val in zip(operations[0].operands, default_block.args, strict=True)
        }
        default_block.add_ops([result := operations[0].clone(value_mapper), YieldOp(result)])
        default_region = Region(default_block)
        # Non-default
        case_regions: list[Region] = []
        if len(operations) > 1:
            for operation in operations[1:]:
                case_block = Block(arg_types=data_operand_types)
                value_mapper = {
                    SSAValue.get(arg): SSAValue.get(val)
                    for arg, val in zip(operation.operands, case_block.args, strict=True)
                }
                case_block.add_ops([result := operation.clone(value_mapper), YieldOp(result)])
                case_regions.append(Region(case_block))
        return ChooseOp(
            name=name,
            data_operands=data_operands,
            switch=switch,
            default_region=default_region,
            case_regions=case_regions,
            result_types=result_types,
        )

    def insert_operations(self, operations: Sequence[Operation]):
        """
        Add an operation to the list of choices if it is not present yet
        """
        # FIXME, what if operation order is swapped? i.e. rhs on lhs side and vice versa?
        data_operand_types = ChooseOp._check_operand_types(self.data_operands, operations)
        for operation in operations:
            if operation.name not in [op.name for op in self.operations()]:
                block = Block(arg_types=data_operand_types)
                block.add_ops([op := type(operation)(*block.args), YieldOp(op)])
                self.add_region(Region(block))

    def operations(self) -> Iterator[Operation]:
        """
        Get an iterator over the list of existing choices of operations
        """
        for region in self.regions:
            # Only yield the first operation in the region
            operation = region.ops.first
            assert operation is not None
            yield operation

    def print(self, printer: Printer):
        printer.print_string(" @")
        printer.print_string(self.name_prop.data)
        printer.print_string(" with ")
        # Print the one switch
        printer.print_operand(self.switch)
        # Print the data operands
        printer.print_string(" (")
        for i, opnd in enumerate(self.data_operands):
            printer.print_operand(opnd)
            printer.print_string(" : ")
            printer.print_attribute(opnd.type)
            if i != len(self.data_operands) - 1:
                printer.print_string(", ")
        printer.print_string(") -> ")
        for i, opnd in enumerate(self.results):
            printer.print_attribute(opnd.type)
            if i != len(self.results) - 1:
                printer.print_string(", ")
        with printer.indented():
            for i, region in enumerate(self.regions):
                printer.print_string(f"\n{i}) (")
                printer.print_list(region.block.args, printer.print_ssa_value)
                printer.print_string(") {")
                printer.print_block(region.block, print_block_args=False)
                printer.print_string("\n}")

    @classmethod
    def parse(cls: type[ChooseOp], parser: Parser) -> ChooseOp:
        name_prop = parser.parse_symbol_name()
        parser.parse_keyword("with")
        switch = parser.parse_operand()

        def parse_itm() -> tuple[SSAValue, Attribute]:
            val = parser.parse_operand()
            parser.parse_punctuation(":")
            typ = parser.parse_type()
            return val, typ

        args: list[tuple[SSAValue, Attribute]] = parser.parse_comma_separated_list(Parser.Delimiter.PAREN, parse_itm)

        parser.parse_punctuation("->")
        res_typ = parser.parse_type()

        def parse_block_arg() -> Parser.UnresolvedArgument:
            val = parser.parse_argument(expect_type=False)
            return val

        def get_choice() -> Region | None:
            if parser.parse_optional_integer() is None:
                return None
            parser.parse_punctuation(")")
            block_args: list[Parser.Argument] = [
                arg.resolve(typ)
                for arg, (_, typ) in zip(
                    parser.parse_comma_separated_list(Parser.Delimiter.PAREN, parse_block_arg), args, strict=True
                )
            ]
            return parser.parse_region(block_args)

        parsed_regions: list[Region] = []
        while True:
            parsed_region = get_choice()
            if parsed_region is not None:
                parsed_regions.append(parsed_region)
            else:
                break

        if len(parsed_regions) == 0:
            parser.raise_error("Expected to parse at least one region!")
        else:
            name = name_prop.data
            data_operands = [el[0] for el in args]
            default_region = parsed_regions[0]
            if len(parsed_regions) > 1:
                case_regions = parsed_regions[1:]
            else:
                case_regions = []
            choose_op = cls(name, data_operands, switch, default_region, case_regions, [res_typ])
            return choose_op


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Operation to select between two inputs.
    The input can be selected with the switch operand.
    """

    name = "phs.mux"

    lhs = operand_def()
    rhs = operand_def()
    switch = operand_def(IndexType)
    res = result_def()

    assembly_format = "`with` $switch ` ``(`$lhs`:`type($lhs)`,` $rhs`:`type($rhs) `)` `->` type($res)  attr-dict"

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        attr_dict: dict[str, Attribute] | None = None,
    ):
        if isinstance(lhs, Operation):
            out_type = lhs.results[0].type
        else:
            out_type = lhs.type
        super().__init__(
            operands=(lhs, rhs, switch),
            attributes=attr_dict,
            # All types should be equal
            result_types=(out_type,),
        )


@irdl_op_definition
class CallOp(IRDLOperation):
    """
    Operation to call a named PEOp operation with data operands and switches.
    """

    name = "phs.call"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    data_operands = var_operand_def()

    switches = var_operand_def(IndexType)
    res = var_result_def()

    traits = traits_def(SymbolOpInterface())
    irdl_options = [AttrSizedOperandSegments()]

    assembly_format = " $sym_name `with` $switches `(`$data_operands type($data_operands)`)` `->` type($res) attr-dict"

    def __init__(
        self,
        name: str,
        data_operands: Sequence[Operation | SSAValue],
        switches: Sequence[Operation | SSAValue],
        result_types: Sequence[Attribute] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            properties={
                "sym_name": StringAttr(name),
            },
            operands=(data_operands, switches),
            attributes=attr_dict,
            result_types=(result_types,),
        )


Phs = Dialect(
    "phs",
    [
        CallOp,
        PEOp,
        MuxOp,
        ChooseOp,
        YieldOp,
    ],
)
