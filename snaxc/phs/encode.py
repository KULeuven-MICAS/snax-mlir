from xdsl.dialects import linalg
from xdsl.dialects.builtin import FunctionType
from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter

from snaxc.dialects import phs


def get_id(op: Operation, count: dict[str, int]):
    """
    Use input and output types to group together operations in similar encoding spaces such that e.g.:
    e.g. the second encountered op with in0 : i32 in1 : i32 and out0 : f32 is be assigned to id
    "i_i32_i32_o_f32"
    """
    key = "i_"
    for opnd in op.operands:
        key += f"{opnd.type}_"

    key += "o_"
    for res in op.results:
        key += f"{res.type}_"

    if key in count:
        current_count = count[key] + 1
        count[key] = current_count
        return key + str(current_count)
    else:
        count[key] = 0
        return key + "0"


def convert_linalg_body_to_phs(linalg_op: linalg.GenericOp, name: str, rewriter: PatternRewriter) -> phs.PEOp:
    """
    Perform conversion from linalg body -> phs body
    """

    count: dict[str, int] = {}

    # Get a copy for conversion of the linalg block
    body_copy = linalg_op.body.clone()
    linalg_yield = body_copy.block.ops.last
    assert isinstance(linalg_yield, linalg.YieldOp)

    pe = phs.PEOp(
        name,
        function_type=FunctionType.from_lists(body_copy.block.arg_types, linalg_yield.operand_types),
        switch_no=0,
        region=body_copy,
    )
    for op in pe.body.ops:
        if isinstance(op, linalg.YieldOp):
            yield_op = phs.YieldOp(op.operands[0])
            rewriter.replace_op(op, yield_op)
        else:
            id = get_id(op, count)
            choose_op = phs.ChooseOp.from_operations(
                id, op.operands, pe.add_switch(), [op], result_types=op.result_types
            )
            rewriter.replace_op(op, choose_op)

    return pe
