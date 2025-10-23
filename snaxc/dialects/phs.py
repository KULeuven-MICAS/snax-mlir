from typing import Iterator, Sequence, Type
from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, Float32Type, IndexType, StringAttr, FunctionType
from xdsl.dialects.func import FuncOpCallableInterface
from xdsl.dialects.utils import AbstractYieldOperation
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
from xdsl.ir import Block, Dialect, Attribute, Region, SSAValue
from xdsl.traits import HasAncestor, IsTerminator, IsolatedFromAbove, Pure, SymbolOpInterface
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
class ChooseOpOp(IRDLOperation):
    name = "phs.choose_op"

    lhs = operand_def(Float32Type)
    rhs = operand_def(Float32Type)

    switch = operand_def(IndexType)

    # cases = prop_def(DenseArrayBase.constr(i64))

    # This is quite similar to a scf.index_switch
    default_region = region_def("single_block")
    case_regions = var_region_def("single_block")

    res = result_def(Float32Type)

    assembly_format = "`(`$lhs`:`type($lhs)`,` $rhs`:`type($rhs)`)` `->` type($res) `with` $switch $default_region $case_regions attr-dict"

    def __init__(
        self,
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        default_region : Region = Region(),
        case_regions : Sequence[Region] = [],
        result_types: Sequence[Attribute] = [],
        attr_dict: dict[str, Attribute] | None = None,
    ):
        super().__init__(
            operands=(lhs, rhs, switch),
            regions=(default_region, case_regions),
            attributes=attr_dict,
            result_types=(result_types,),
        )

    @staticmethod
    def from_operation(
        lhs: Operation | SSAValue,
        rhs: Operation | SSAValue,
        switch: Operation | SSAValue,
        operation: type[FloatingPointLikeBinaryOperation],
        result_types: Sequence[Attribute] = [],
    ):
        return ChooseOpOp(
            lhs=lhs,
            rhs=rhs,
            switch=switch,
            default_region=
                Region(
                    Block([
                        result:=operation(lhs, rhs),
                        YieldOp(result),
                ])),
            result_types=result_types
        )

    def add_operation(self, operation: type[FloatingPointLikeBinaryOperation]):
        """
        Add an operation to the list of choices
        """
        self.add_region(Region(Block([op := operation(self.lhs, self.rhs), YieldOp(op)])))

    def operations(self) -> Iterator[FloatingPointLikeBinaryOperation]:
        """
        Get an iterator over the list of existing choices of operations
        """
        for region in self.regions:
            # Only yield the first operation in the region
            operation = region.ops.first
            assert isinstance(operation, FloatingPointLikeBinaryOperation)
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


@irdl_op_definition
class AbstractPEOperation(IRDLOperation):
    name = "phs.abstract_pe"

    name_prop = prop_def(StringAttr, prop_name="sym_name")

    function_type = prop_def(FunctionType)

    body = region_def("single_block")

    function_type = prop_def(FunctionType)
    arg_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])
    res_attrs = opt_prop_def(ArrayAttr[DictionaryAttr])

    traits = traits_def(IsolatedFromAbove(), SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(
        self,
        name : str,
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
            raise VerifyException(
                "Expected entry block arguments to have the same types as the function "
                "input types"
            )

    def walk_choose_ops(self, *, reverse :bool, region_first : bool) -> Iterator[Operation]:
        for op in self.walk(reverse=reverse, region_first=region_first):
            if isinstance(op, ChooseOpOp):
                yield op


Phs = Dialect(
    "phs",
    [
        AbstractPEOperation,
        ChooseInputOp,
        ChooseOpOp,
        YieldOp,
    ],
)
