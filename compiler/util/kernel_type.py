from xdsl.dialects import linalg, arith
from xdsl.dialects.builtin import IntegerType, ShapedType
from xdsl.ir.core import Block
from enum import Enum
from typing import List


class KernelType(Enum):
    MUL = "mul"
    MAC = "mac"
    QMAC = "qmac"

    @staticmethod
    def get_type(linalg_block: Block, input_types: List):
        # detect the kernel type of a linalg block
        # will detect either MUL, MAC, or QMAC
        # if it detects neither of these ops, the
        # function returns none

        # generate a dictionary of the input types of the linalg
        # (the inputs of the linalg block are all i8 or i32,
        # but is interesing if the input of the actual linalg.generic
        # is a tensor or an integer type)
        types = dict(zip(linalg_block.args, input_types))

        # block should work on three shaped arguments (in_0, in_1, out)
        # possibly some other constants (zero-point offsets)

        # this check therefore counts the number of inputs which are shaped inputs
        # other inputs may be integer types used for constant operations, such as
        # zero-point offsets
        if len([type for type in input_types if isinstance(type, ShapedType)]) != 3:
            return None

        # last op of the linalg block should be a yield op
        if not isinstance(linalg_block.last_op, linalg.YieldOp):
            return None

        # go to producing op of yield
        op = linalg_block.last_op.arguments[0].op

        # if the last op is a multiplication, this kernel may be a mult op
        # this is true if the inputs of the muli are directly the inputs of
        # the linalg block
        if isinstance(op, arith.Muli):
            if (
                linalg_block.args[0] in op.operands
                and linalg_block.args[1] in op.operands
            ):
                return KernelType.MUL
            return None

        # if the last op is an addition, this may be multiply-accumulate kernel
        if isinstance(op, arith.Addi):
            # for this to be an accumulate operation, one of the operands
            # must be the output operand (last arg) of the linalg block
            if op.lhs is linalg_block.args[-1]:
                mult_op = op.rhs.op
            elif op.rhs is linalg_block.args[-1]:
                mult_op = op.lhs.op
            else:
                return None

            # the other operand must be the multiplication of the two inputs
            if not isinstance(mult_op, arith.Muli):
                return None

            # The following block is an algorithm to determine if the two inputs
            # of the multiplication operation root from the linalg inputs, there
            # may be some legal operations on this operand, such as a number of
            # sign extensions or zero-point adjustments, the algorithm will go
            # through all the operations until we reach the input of the linalg
            # block. If there are unknown operations encountered, we return None

            # flag to keep track if a zero-point adjustment has been made
            has_zero_point_adjustment = False
            # set to keep track of used tensor inputs:
            # get first two shaped inputs of linalg block, these are the tensor inputs
            linalg_block_inputs = set(
                [
                    inp
                    for inp in linalg_block.args[0:-1]
                    if isinstance(types[inp], ShapedType)
                ]
            )

            # do the check for both operands of the multiplication
            for op in [mult_op.lhs.op, mult_op.rhs.op]:
                while True:
                    if isinstance(op, arith.ExtSIOp):
                        # signed extensions are allowed
                        if op.operands[0] in linalg_block_inputs:
                            # reached root op, break loop
                            op = op.operands[0]
                            linalg_block_inputs.remove(op)
                            break
                        if hasattr(op.operands[0], "op"):
                            # else continue
                            op = op.operands[0].op
                            continue
                        return None
                    if isinstance(op, arith.Subi):
                        # subi operations may be the result of a zero-point
                        # adjustment, for this,
                        # one of the operands must be constant
                        if op.lhs in linalg_block.args and isinstance(
                            types[op.lhs], IntegerType
                        ):
                            if op.rhs in linalg_block_inputs:
                                # reached root op, break loop
                                op = op.rhs
                                linalg_block_inputs.remove(op)
                                has_zero_point_adjustment = True
                                break
                            if hasattr(op.rhs, "op"):
                                # else continue
                                op = op.rhs.op
                                continue
                            return None
                        elif op.rhs in linalg_block.args and isinstance(
                            types[op.rhs], IntegerType
                        ):
                            if op.lhs in linalg_block_inputs:
                                # reached root
                                op = op.lhs
                                linalg_block_inputs.remove(op)
                                has_zero_point_adjustment = True
                            if hasattr(op.lhs, "op"):
                                # else continue
                                op = op.lhs.op
                                continue
                            return None
                        else:
                            # subtraction without a constant argument,
                            # thus not a supported operation
                            return None
                    # not a supported op found in the kernel, thus return None
                    return None
            if has_zero_point_adjustment:
                return KernelType.QMAC
            else:
                return KernelType.MAC
