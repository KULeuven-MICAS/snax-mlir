from typing import Iterator, Sequence, Generator
from xdsl import dialects
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, Float32Type, IndexType, StringAttr, FunctionType
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects.utils import AbstractYieldOperation
from xdsl.dialects import linalg
from xdsl.irdl import (
    Operation,
    irdl_op_definition,
    IRDLOperation,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    region_def,
    traits_def,
    var_region_def,
    lazy_traits_def,
)
from xdsl.ir import Block, BlockArgument, Dialect, Attribute, Region, SSAValue
from xdsl.parser import SymbolRefAttr
from xdsl.traits import HasAncestor, HasParent, IsTerminator, IsolatedFromAbove, Pure, SymbolOpInterface, SymbolTable
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.arith import FloatingPointLikeBinaryOperation


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
    def from_operation(acc_ref: SymbolRefAttr, operation: FloatingPointLikeBinaryOperation) -> "AbstractPEOperation":
        """
        Utility constructor that fills up an Abstract PE operation with a simple preset
        based on an operation
        """
        switch_types = [IndexType()]
        # Based on operation
        in_types = [operation.lhs.type, operation.rhs.type]
        out_types = [operation.result.type]
        # Construct a new block based on the input of the
        block_inputs = [*in_types, *switch_types]
        block = Block(arg_types=block_inputs)
        # Map block args to inputs and outputs to yield
        lhs, rhs, switch = block.args
        block.add_ops(
            [
                result := ChooseOpOp.from_operation(
                    "0", lhs, rhs, switch, operation=type(operation), result_types=out_types
                ),
                YieldOp(result),
            ]
        )
        abstract_pe_op = AbstractPEOperation(acc_ref.string_value(), (block_inputs, out_types), Region(block))
        return abstract_pe_op


    def get_choose_op(self, symbol_name : str ) -> "ChooseOpOp | None":
        t = self.get_trait(SymbolTable)
        assert t is not None
        choose_op_op = t.lookup_symbol(self, symbol_name)
        if choose_op_op is None:
            return choose_op_op
        else:
            assert isinstance(choose_op_op, ChooseOpOp)
            return choose_op_op

  #  @staticmethod
  #  def from_generic_body(acc_ref: SymbolRefAttr, generic_body: Block) -> "AbstractPEOperation":
  #      """
  #      Add operations from a linalg generic body to the Abstract PE
  #      """
  #      linalg_yield = generic_body.ops.last
  #      # Error if not a linalg body
  #      assert isinstance(linalg_yield, linalg.YieldOp), 'No linalg.yield found, is this a valid linalg.generic body?'
  #      # Error if multiple values are yielded
  #      assert len(linalg_yield.results) == 1
  #      linalg_yield.results[]
  #      return None




    def _add_extra_switch(self) -> BlockArgument:
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

    assembly_format = " $sym_name` ` `(`$lhs`:`type($lhs)`,` $rhs`:`type($rhs)`)` `->` type($res) `with` $switch $default_region $case_regions attr-dict"

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

    @staticmethod
    def from_operation(
        name: str,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        operation: type[FloatingPointLikeBinaryOperation],
        result_types: Sequence[Attribute] = [],
    ) -> "ChooseOpOp":
        return ChooseOpOp(
            name=name,
            lhs=lhs,
            rhs=rhs,
            switch=switch,
            default_region=Region(
                Block(
                    [
                        result := operation(lhs, rhs),
                        YieldOp(result),
                    ]
                )
            ),
            result_types=result_types,
        )

    def add_operation(self, operation: type[FloatingPointLikeBinaryOperation]):
        """
        Add an operation to the list of choices
        """
        self.add_region(Region(Block([op := operation(self.lhs, self.rhs), YieldOp(op)])))

    def operations(self) -> Iterator[FloatingPointLikeBinaryOperation | None]:
        """
        Get an iterator over the list of existing choices of operations
        """
        for region in self.regions:
            # Only yield the first operation in the region
            operation = region.ops.first
            if isinstance(operation, FloatingPointLikeBinaryOperation):
                yield operation
            else:
                yield None


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
