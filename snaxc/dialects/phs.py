from collections.abc import Iterator, Sequence

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
    var_region_def,
)
from xdsl.parser import SymbolRefAttr
from xdsl.traits import HasAncestor, HasParent, IsolatedFromAbove, IsTerminator, Pure, SymbolOpInterface, SymbolTable
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class YieldOp(AbstractYieldOperation[Attribute]):
    name = "phs.yield"

    traits = lazy_traits_def(
        lambda: (
            IsTerminator(),
            HasAncestor(AbstractPEOperation),
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
class AbstractPEOperation(IRDLOperation):
    name = "phs.abstract_pe"

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
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
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
    def from_operations(
        acc_ref: SymbolRefAttr, operations: Sequence[FloatingPointLikeBinaryOperation]
    ) -> "AbstractPEOperation":
        """
        Utility constructor that fills up an Abstract PE operation with a simple preset
        based on an operation
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
                result := ChooseOpOp.from_operations(
                    "0", lhs, rhs, switch, operations=type_ops, result_types=out_types
                ),
                YieldOp(result),
            ]
        )
        abstract_pe_op = AbstractPEOperation(acc_ref.string_value(), (block_inputs, out_types), Region(block))
        return abstract_pe_op

    def get_choose_op(self, symbol_name: str) -> "ChooseOpOp | None":
        t = self.get_trait(SymbolTable)
        assert t is not None, "No SymbolTable present in current operation"
        choose_op_op = t.lookup_symbol(self, symbol_name)
        if choose_op_op is None:
            return choose_op_op
        else:
            assert isinstance(choose_op_op, ChooseOpOp)
            return choose_op_op

    def get_terminator(self) -> YieldOp:
        yield_op = self.body.ops.last
        assert isinstance(yield_op, YieldOp)
        return yield_op

    def add_extra_switch(self) -> BlockArgument:
        block = self.regions[0].blocks.first
        assert block is not None
        # Add new switch at the end
        self.function_type = FunctionType.from_lists(
            list(self.function_type.inputs) + [IndexType()], list(self.function_type.outputs)
        )
        return block.insert_arg(IndexType(), len(block.args))

    def walk_choose_ops(self, *, reverse: bool = False, region_first: bool = False) -> "Iterator[ChooseOpOp]":
        for op in self.walk(reverse=reverse, region_first=region_first):
            if isinstance(op, ChooseOpOp):
                yield op


@irdl_op_definition
class ChooseOpOp(IRDLOperation):
    name = "phs.choose_op"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    lhs = operand_def(Float32Type)
    rhs = operand_def(Float32Type)

    switch = operand_def(IndexType)

    # cases = prop_def(DenseArrayBase.constr(i64))

    # This is quite similar to a scf.index_switch
    default_region = region_def("single_block")
    case_regions = var_region_def("single_block")

    res = result_def(Float32Type)

    assembly_format = (
        " $sym_name` ` `(`$lhs`:`type($lhs)`,` $rhs`:`type($rhs)`)` `->`"
        + " type($res) `with` $switch $default_region $case_regions attr-dict"
    )

    traits = traits_def(SymbolOpInterface(), HasParent(AbstractPEOperation))

    def __init__(
        self,
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        default_region: Region = Region(Block([YieldOp()])),
        case_regions: Sequence[Region] = [],
        result_types: Sequence[Attribute] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        # FIXME if you use an empty region this thing fails verification
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
        """Returns all operands except the switch."""
        return self._operands[:-1]

    @staticmethod
    def from_operations(
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        operations: Sequence[type[FloatingPointLikeBinaryOperation]],
        result_types: Sequence[Attribute] = [],
    ) -> "ChooseOpOp":
        # Default operation
        case_regions: list[Region] = []
        if len(operations) < 1:
            for operation in operations[1:]:
                case_regions.append(
                    Region(
                        Block(
                            [
                                result := operation(lhs, rhs),
                                YieldOp(result),
                            ]
                        )
                    )
                )
        default_region = Region(
            Block(
                [
                    result := operations[0](lhs, rhs),
                    YieldOp(result),
                ]
            )
        )
        return ChooseOpOp(
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


@irdl_op_definition
class ChooseInputOp(IRDLOperation):
    name = "phs.choose_input"

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


Phs = Dialect(
    "phs",
    [
        AbstractPEOperation,
        ChooseInputOp,
        ChooseOpOp,
        YieldOp,
    ],
)
