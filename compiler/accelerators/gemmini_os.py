from typing import Sequence
from compiler.accelerators.gemmini import GemminiAccelerator
from xdsl.dialects import linalg
from xdsl.ir import Operation

class GemminiMvinAccelerator(GemminiAccelerator):
    name = "gemmini_mvin"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch register sets.
    Also, there are actually 3 data movers, the docs say this is to use them
    in parallel, but they are not programmed in parallel in the OS kernel?
    Hence the name k_CONFIG (opcode) CONFIG_LD (modifier for loading/mvin)
    and id0 (modifier for which of 3 movers is used).

    Note: You need to manually make sure: 
    * that id is programmed to cst=0!
    * that CONFIG_LD modifier is set for k_CONFIG opcode
    """
    fields = {
        "k_CONFIG_k_CONFIG_LD_id0.rs1": 0,
        "k_CONFIG_k_CONFIG_LD_id0.rs2": 0,
    }
    launch_fields = {
        "k_MVIN.rs1": 2,
        "k_MVIN.rs2": 2,
    }


class GemminiMvoutAccelerator(GemminiAccelerator):
    name = "gemmini_mvout"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch registers sets.
    Hence the name k_CONFIG (opcode) CONFIG_ST (modifier for storing/mvout)

    Note: You need to manually make sure: 
    * that CONFIG_ST modifier is set for k_CONFIG opcode
    """
    fields = {
        "k_CONFIG_k_CONFIG_ST.rs1": 0,
        "k_CONFIG_k_CONFIG_ST.rs2": 0,
    }
    launch_fields = {
        "k_MVOUT.rs1": 3,
        "k_MVOUT.rs2": 3,
    }

class GemminiExAccelerator(GemminiAccelerator):
    name = "gemmini_ex"
    """
    For some weird reason, all of these config instructions use the same 
    k_CONFIG opcode, but they use one of rs's contents to switch registers sets.
    Hence the name k_CONFIG (opcode) CONFIG_EX (modifier for execution)

    Note: You need to manually make sure: 
    * that CONFIG_EX modifier is set for k_CONFIG opcode
    * the launch fields of the final instruction in this sequence are the same,
      but the compiler needs to emit both accumulating and non-accumulating
      instructions, because this is based on the op-code of the operation.
    """
    fields = {
        "k_CONFIG_k_CONFIG_EX.rs1": 0,
        "k_CONFIG_k_CONFIG_EX.rs2": 0,
    }
    launch_fields = {
        "k_PRELOAD.rs1": 2,
        "k_PRELOAD.rs2": 2,
    }

def convert_to_accfg_sequence(op : linalg.Generic) -> Sequence[Operation]:
    return None
