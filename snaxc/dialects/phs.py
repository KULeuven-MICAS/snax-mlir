from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

from xdsl.dialects.arith import FloatingPointLikeBinaryOperation
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, Float32Type, FunctionType, IndexType, StringAttr
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.ir import Attribute, Block, BlockArgument, Dialect, Region, SSAValue
from xdsl.irdl import (
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
)
from xdsl.parser import Parser, SymbolRefAttr
from xdsl.printer import Printer
from xdsl.traits import HasAncestor, HasParent, IsolatedFromAbove, IsTerminator, Pure, SymbolOpInterface, SymbolTable
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
            HasAncestor(PEOp),
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
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface(), SymbolTable())

    def __init__(
        self,
        name: str,
        function_type: FunctionType | tuple[Sequence[Attribute], Sequence[Attribute]],
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
                "arg_attrs": arg_attrs,
                "res_attrs": res_attrs,
            },
            regions=[region],
        )

    def verify_(self) -> None:
        # If this is an empty region
        if len(self.body.blocks) == 0:
            return

        entry_block = self.body.blocks.first
        assert entry_block is not None
        block_arg_types = entry_block.arg_types
        if self.function_type.inputs.data != tuple(block_arg_types):
            raise VerifyException("Expected entry block arguments to have the same types as the function input types")

    @staticmethod
    def from_operations(acc_ref: SymbolRefAttr, operations: Sequence[FloatingPointLikeBinaryOperation]) -> PEOp:
        """
        Utility constructor that fills up an Abstract PE operation with a simple preset
        based on a sequence of operations
        """
        switch_types = [IndexType()]
        # Based on operation
        # FIXME check if all operations have the same types!
        in_types = [operations[0].lhs.type, operations[0].rhs.type]
        out_types = [operations[0].result.type]
        # Construct a new block based on the input of the
        block_inputs = [*in_types, *switch_types]
        block = Block(arg_types=block_inputs)
        # Map block args to inputs and outputs to yield
        lhs, rhs, switch = block.args
        type_ops = [type(op) for op in operations]
        block.add_ops(
            [
                result := ChooseOp.from_operations("0", lhs, rhs, switch, operations=type_ops, result_types=out_types),
                YieldOp(result),
            ]
        )
        pe_op = PEOp(acc_ref.string_value(), (block_inputs, out_types), Region(block))
        return pe_op

    def get_choose_op(self, symbol_name: str) -> ChooseOp | None:
        """
        Get a specific choose op inside the AbstractPEOp by symbol name
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
        Get the terminating operation inside the AbstractPEOp body.
        """
        yield_op = self.body.ops.last
        assert isinstance(yield_op, YieldOp)
        return yield_op

    def add_switch(self) -> BlockArgument:
        """
        Add an extra switch to the Abstract PE operation
        """
        block = self.regions[0].blocks.first
        assert block is not None
        # Add new switch at the end
        self.function_type = FunctionType.from_lists(
            list(self.function_type.inputs) + [IndexType()], list(self.function_type.outputs)
        )
        return block.insert_arg(IndexType(), len(block.args))


@irdl_op_definition
class ChooseOp(IRDLOperation):
    """
    Operation to choose between operations contained in its region.
    Very similar to scf.index_switch.
    """

    name = "phs.choose"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    lhs = operand_def(Float32Type)
    rhs = operand_def(Float32Type)

    switch = operand_def(IndexType)

    # This is quite similar to a scf.index_switch
    default_region = region_def("single_block")
    case_regions = var_region_def("single_block")

    res = result_def(Float32Type)

    traits = traits_def(SymbolOpInterface(), HasParent(PEOp))

    def __init__(
        self,
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
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
            operands=(lhs, rhs, switch),
            regions=(default_region, case_regions),
            attributes=attr_dict,
            result_types=(result_types,),
        )

    @property
    def data_operands(self) -> Sequence[SSAValue]:
        """
        Returns all operands except the switch operands
        """
        return self._operands[:-1]

    @staticmethod
    def from_operations(
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        operations: Sequence[type[FloatingPointLikeBinaryOperation]],
        result_types: Sequence[Attribute] = [],
    ) -> ChooseOp:
        """
        Utility constructor to construct a ChooseOp from a set of given operations
        """
        # Default operation
        case_regions: list[Region] = []
        if len(operations) > 1:
            for operation in operations[1:]:
                case_regions.append(Region(Block([result := operation(lhs, rhs), YieldOp(result)])))
        default_region = Region(Block([result := operations[0](lhs, rhs), YieldOp(result)]))
        return ChooseOp(
            name=name,
            lhs=lhs,
            rhs=rhs,
            switch=switch,
            default_region=default_region,
            case_regions=case_regions,
            result_types=result_types,
        )

    def insert_operations(self, operations: Sequence[FloatingPointLikeBinaryOperation]):
        """
        Add an operation to the list of choices if it is not present yet
        """
        # FIXME, what if operation order is swapped? i.e. rhs on lhs side and vice versa?
        for operation in operations:
            if operation.name not in [op.name for op in self.operations()]:
                self.add_region(Region(Block([op := type(operation)(self.lhs, self.rhs), YieldOp(op)])))

    def operations(self) -> Iterator[FloatingPointLikeBinaryOperation]:
        """
        Get an iterator over the list of existing choices of operations
        """
        for region in self.regions:
            # Only yield the first operation in the region
            operation = region.ops.first
            if isinstance(operation, FloatingPointLikeBinaryOperation):
                yield operation

    def print(self, printer: Printer):
        printer.print_string(" @")
        printer.print_string(self.name_prop.data)
        printer.print_string(" with ")
        printer.print_operand(self.operands[-1])
        printer.print_string(" (")
        for i, opnd in enumerate(self.operands[:-1]):
            printer.print_operand(opnd)
            printer.print_string(" : ")
            printer.print_attribute(opnd.type)
            if i != len(self.operands) - 2:
                printer.print_string(", ")
        printer.print_string(") -> ")
        for i, opnd in enumerate(self.results):
            printer.print_attribute(opnd.type)
            if i != len(self.results) - 1:
                printer.print_string(", ")
        printer.print_string(" {")
        with printer.indented():
            for i, region in enumerate(self.regions):
                printer.print_string(f"\n{i}) ")
                # FIXME, what if multiple operations?
                for op in list(region.block.ops)[:-1]:
                    printer.print_string(op.name)
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
        assert len(args) == 2, "Expect to have lhs and rhs operand for ChooseOp"

        parser.parse_comma_separated_list
        parser.parse_punctuation("->")
        res_typ = parser.parse_type()
        parser.parse_punctuation("{")

        def get_op() -> type[Operation] | None:
            if parser.parse_optional_integer() is None:
                return None
            parser.parse_punctuation(")")
            operation_ident = parser.parse_identifier()
            # FIXME is there a public thing I can use to do this?
            return parser._get_op_by_name(operation_ident)  # pyright: ignore [reportPrivateUsage]

        parsed_operations: list[type[Operation]] = []
        while True:
            parsed_operation = get_op()
            if parsed_operation is not None:
                parsed_operations.append(parsed_operation)
            else:
                break

        assert len(parsed_operations) >= 1, "Expected to parse at least one operation!"
        parser.parse_punctuation("}")
        for operation in parsed_operations:
            assert issubclass(operation, FloatingPointLikeBinaryOperation)
        typed_operations = cast(Sequence[type[FloatingPointLikeBinaryOperation]], parsed_operations)
        choose_op = cls.from_operations(name_prop.data, args[0][0], args[1][0], switch, typed_operations, [res_typ])
        return choose_op


@irdl_op_definition
class MuxOp(IRDLOperation):
    """
    Operation to select between two inputs.
    The input can be selected with the switch operand.
    """

    name = "phs.mux"

    lhs = operand_def(Float32Type)
    rhs = operand_def(Float32Type)
    switch = operand_def(IndexType)
    res = result_def(Float32Type)

    assembly_format = "`(`$lhs`:`type($lhs)`,` $rhs`:`type($rhs) `)` `->` type($res) `with` $switch attr-dict"

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        result_types: Sequence[Attribute],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=(lhs, rhs, switch),
            attributes=attr_dict,
            result_types=(result_types,),
        )


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "phs.call"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    lhs = operand_def(Float32Type)
    rhs = operand_def(Float32Type)
    switches = var_operand_def(IndexType)
    res = result_def(Float32Type)

    traits = traits_def(SymbolOpInterface())

    def __init__(
        self,
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switches: Sequence[Operation | SSAValue],
        result_types: Sequence[Attribute] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            properties={
                "sym_name": StringAttr(name),
            },
            operands=(lhs, rhs, switches),
            attributes=attr_dict,
            result_types=(result_types,),
        )


Phs = Dialect(
    "phs",
    [
        PEOp,
        MuxOp,
        ChooseOp,
        CallOp,
        YieldOp,
    ],
)
