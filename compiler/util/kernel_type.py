from enum import Enum

from xdsl.dialects import arith, linalg
from xdsl.dialects.builtin import ShapedType
from xdsl.ir import Block, Operation, OpResult, SSAValue


class KernelException(Exception):
    """
    A KernelException is raised by the parsing functions if the provided
    operation does not match the expected operation type in the kernel.
    """

    pass


class InputException(KernelException):
    """
    A InputException is a specific case of parsing error, when we fail to
    match given operands to the inputs of the kernel. The current parsing
    strategy is therefore not necessarily incorrect, just incomplete.
    """

    pass


class KernelType(Enum):
    """
    Enumeration representing different types of kernel operations of a
    linalg generic body.

    Attributes:
        MUL (str): Represents a multiplication operation.
            out = a * b
        MAC (str): Represents a multiply-accumulate operation.
            out += a * b
        QMAC (str): Represents a quantized multiply-accumulate operation.
            out += (a - zp_a) * (b - zp_b)
    """

    MUL = "mul"
    MAC = "mac"
    QMAC = "qmac"

    @staticmethod
    def parse_mult(op: Operation) -> tuple[Operation, Operation]:
        """
        Parses a multiplication operation and returns the two operands.

        Args:
            op (Operation): The multiplication operation.

        Returns:
            tuple[Operation, Operation]: The two operands of the
            multiplication operation.

        Raises:
            KernelException: If the provided operation is not a multiplication.
        """

        # check if op is a multiplication
        if not isinstance(op, arith.Muli | arith.Mulf):
            raise KernelException("not a multiplication")

        # get operands
        a = op.lhs
        b = op.rhs
        return a, b

    @staticmethod
    def parse_add(op: Operation) -> tuple[Operation | SSAValue, Operation | SSAValue]:
        """
        Parses an addition operation and returns the two operands.

        Args:
            op (Operation): The addition operation.

        Returns:
            tuple[Operation | SSAValue, Operation | SSAValue]: The two operands
            of the addition operation.

        Raises:
            KernelException: If the provided operation is not an addition.
        """

        # check if op is a multiplication
        if not isinstance(op, arith.Addi | arith.Addf):
            raise KernelException("not an addition")

        # get operands
        a = op.lhs
        b = op.rhs
        return a, b

    @staticmethod
    def parse_zpa(op: Operation) -> tuple[Operation | SSAValue, Operation | SSAValue]:
        """
        Parses a zero point adjustment operation and returns the two operands.
        A zero point adjustment operation is a subtraction operation, where
        the right operand is a zero point offset and the left operand is the
        input tensor. Before the subtraction, the input tensor may be sign
        extended to avoid overflow.

        Args:
            op (Operation): The subtraction operation.

        Returns:
            tuple[Operation | SSAValue, Operation | SSAValue]: The two operands
            of the subtraction operation.

        Raises:
            KernelException: If the provided operation is not a subtraction.
        """

        # check if op is a subtraction
        if not isinstance(op, arith.Subi):
            raise KernelException

        # get operands
        value = op.lhs
        adjustment = op.rhs

        # additional sign extension possible
        while isinstance(value, OpResult) and isinstance(value.op, arith.ExtSIOp):
            value = value.op.operands[0]

        return value, adjustment

    @staticmethod
    def parse_inputs(linalg_op: linalg.Generic) -> tuple[linalg.YieldOp, dict]:
        """
        Parses the inputs of a linalg operation and returns the yield operation
        of the linalg body and a dictionary of operand types. The dictionary of
        operand types is used to match the block arguments (scalars) to the actual
        operand types of the linalg.Generic (shaped memrefs).

        Args:
            linalg_op (linalg.Generic): The linalg operation.

        Returns:
            tuple[linalg.YieldOp, dict]: The yield operation and a
            dictionary of operand types.

        Raises:
            KernelException: If the number of shaped inputs is incorrect for binary ops.
            KernelException: If number of outputs is incorrect for binary ops.
        """

        linalg_block: Block = linalg_op.body.block
        operand_types: list = [o.type for o in linalg_op.operands]

        # the operand of the linalg block are scalars, but to detect
        # the kernel type, we need to know if the operand of the linalg
        # block are tensors or scalars, therefore we generate a dictionary
        # of the input types of the linalg
        types = dict(zip(linalg_block.args, operand_types))

        # this check counts the number of operand which are shaped operand
        # for all the kernels there may be three shaped operands
        # (input a, input b, output c)  other operands may be integer types used for
        # constant values, such as zero-point offsets
        if len([type for type in types.values() if isinstance(type, ShapedType)]) != 3:
            raise KernelException("Wrong number of shaped operands")

        # there should only be one output and it should be shaped
        if len(linalg_op.outputs) != 1:
            raise KernelException("Wrong number of outputs")

        # input parsing successful, we can return the computed types
        # for further checking
        return (linalg_block.last_op, types)

    @staticmethod
    def match_inputs(a: Operation, b: Operation, types: dict):
        """
        Matches the operands a and b with the input types of the linalg kernel.
        This is mainly used as a check to see if we have reached the top
        of the computation graph in the linalg kernel.

        Args:
            a (Operation): The first operand.
            b (Operation): The second operand.
            types (dict): The dictionary of input types.

        Raises:
            InputException: If operand a is not a shaped input.
            InputException: If operand b is not a shaped input.
        """
        # check if the operands a and b are shaped inputs of the linalg kernel
        if a not in types or not isinstance(types[a], ShapedType):
            raise InputException("Operand a is not a shaped input")
        if b not in types or not isinstance(types[b], ShapedType):
            raise InputException("Operand b is not a shaped input")

        # matching successful, return
        return

    @staticmethod
    def get_kernel(linalg_op: linalg.Generic):
        """
        Determines the kernel type of a linalg operation.

        Args:
            linalg_op (linalg.Generic): The linalg operation.

        Returns:
            KernelType | None: The kernel type if it matches any of the
            supported types, None otherwise.
        """

        try:
            yield_op, types = KernelType.parse_inputs(linalg_op)
        except KernelException:
            return None

        # make sure yield_op is produced by other op before parsing
        # if not, the kernel is empty (just linalg yield), and
        # yielded_op.op would result in an AttributeError
        yielded_op = yield_op.operands[0]
        if not isinstance(yielded_op, OpResult):
            return None

        # check: MUL
        # a = b * c
        try:
            val_a, val_b = KernelType.parse_mult(yielded_op.op)
            KernelType.match_inputs(val_a, val_b, types)
            return KernelType.MUL
        except KernelException:
            pass

        # check: MAC \ QMAC
        # a += c * d (or a += (c - zp_c) * (d - zp_d))
        try:
            val_a, val_b = KernelType.parse_add(yielded_op.op)
            # one of the values is the accumulation
            # (the last argument of the linalg block),
            # the other one is the multiplication of c and d
            accumulation = linalg_op.body.block.args[-1]
            mult_op = val_b if val_a is accumulation else val_a
            if not isinstance(mult_op, OpResult):
                raise KernelException
            mult_op = mult_op.op

            val_c, val_d = KernelType.parse_mult(mult_op)

            try:
                # for successful input match, return MAC
                KernelType.match_inputs(val_c, val_d, types)
                return KernelType.MAC

            except InputException:
                # maybe additional zero-point adjustment for QMAC
                if not isinstance(val_c, OpResult) or not isinstance(val_d, OpResult):
                    raise KernelException
                val_c_zp, zp_c = KernelType.parse_zpa(val_c.op)
                val_d_zp, zp_d = KernelType.parse_zpa(val_d.op)
                KernelType.match_inputs(val_c_zp, val_d_zp, types)
                return KernelType.QMAC
        except KernelException:
            pass

        return None
