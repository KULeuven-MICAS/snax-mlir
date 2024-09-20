import re

CYCLE_REGEX = re.compile(
    r"\s*([0-9]+) ([0-9]+)\s+[0-9]+\s+(0x[0-9a-fz]+)\s+DASM\(([0-9a-fz]+)\)\s*#;(.*)"
)


class Instruction:
    raw_encoding: int

    __slots__ = ["raw_encoding"]

    def __new__(cls, raw_encoding: int) -> "Instruction":
        if cls is not Instruction:
            instance = super().__new__(cls)
            instance.raw_encoding = raw_encoding
            return instance

        opcode = raw_encoding & 0x7F
        for subclass in cls.__subclasses__():
            if opcode in subclass.OP_CODES:
                return subclass(raw_encoding)

        instance = super().__new__(cls)
        instance.raw_encoding = raw_encoding
        return instance

    @property
    def opcode(self) -> int:
        return self.raw_encoding & 0x7F


class CSRInstruction(Instruction):
    OP_CODES = {0b1110011}

    @property
    def funct3(self) -> int:
        return (self.raw_encoding >> 12) & 0b111

    @property
    def is_csrrsi(self):
        return self.funct3 == 0b110

    @property
    def is_csrrci(self):
        return self.funct3 == 0b111

    @property
    def csr(self):
        return self.raw_encoding >> 20


class RInstruction(Instruction):
    OP_CODES = {0x2B}

    def __new__(cls, raw_encoding: int) -> "RInstruction":
        if cls is not RInstruction:
            return super().__new__(cls, raw_encoding)

        r_inst = super().__new__(cls, raw_encoding)
        for subclass in cls.__subclasses__():
            if (
                r_inst.opcode == subclass.OPCODE
                and r_inst.funct3 == subclass.FUNCT3
                and r_inst.funct7 == subclass.FUNCT7
            ):
                return subclass(raw_encoding)

        return r_inst

    @classmethod
    def read_rs1(cls, state: "TraceState"):
        return state.cpu_state["opa"]

    @classmethod
    def read_rs2(cls, state: "TraceState"):
        return state.cpu_state["opb"]

    @property
    def rs2_imm5(self):
        return (self.raw_encoding >> 20) & 0x1F

    @property
    def funct3(self) -> int:
        return (self.raw_encoding >> 12) & 0x7

    @property
    def funct7(self) -> int:
        return (self.raw_encoding >> 25) & 0x7F


class DMSRCInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0

    read_source = RInstruction.read_rs1


class DMDSTInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0b1

    read_destination = RInstruction.read_rs1


class DMREPInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0b111

    read_reps = RInstruction.read_rs1


class DMSTRInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0b110

    read_source_strides = RInstruction.read_rs1
    read_dest_strides = RInstruction.read_rs2


class DMCPYIInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0b10

    read_size = RInstruction.read_rs1
    config = RInstruction.rs2_imm5

    @property
    def is_2d(self):
        return (self.config & 0b10) != 0


class DMSTATIInstruction(RInstruction):
    OPCODE = 0x2B
    FUNCT3 = 0
    FUNCT7 = 0b100

    status = RInstruction.rs2_imm5


def get_trace_state(line: str):
    match = CYCLE_REGEX.match(line)
    if match is None:
        return None

    pc = match.group(3)
    # No program counter if we are within the sequencer.
    if pc == "0xzzzzzzzz":
        pc = None
    else:
        pc = int(pc, base=16)

    op_code = match.group(4)
    # freps in the sequencer don't have an opcode.
    if op_code == "zzzzzzzzzzzzzzzz":
        op_code = None
    else:
        op_code = int(op_code, base=16)

    return TraceState(int(match.group(2), base=10), pc, op_code, eval(match.group(5)))


class TraceState:
    clock_cycle: int
    # PC or None if in FPSS sequencer.
    pc: int | None
    # None if an frep.
    instruction: Instruction | None
    cpu_state: dict

    def __init__(self, clock_cycle, pc, op_code, cpu_state):
        self.clock_cycle = clock_cycle
        self.pc = pc
        self.instruction = None if op_code is None else Instruction(op_code)
        self.cpu_state = cpu_state

    @property
    def in_fpss_sequencer(self):
        return self.pc is None
