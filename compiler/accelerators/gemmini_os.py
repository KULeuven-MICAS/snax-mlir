from typing import Sequence
from compiler.accelerators.gemmini import GemminiAccelerator
from compiler.dialects import accfg
from xdsl.dialects import linalg, arith
from xdsl.ir import Operation, SSAValue
from abc import ABC, abstractmethod

class GemminiOsAcceleratorBase(GemminiAccelerator, ABC):
    """
    Abstract base class for Gemmini accelerator instances in
    output stationary mode
    """

    def get_setup_op(self, vals : Sequence[Operation | SSAValue]) -> accfg.SetupOp:
        return accfg.SetupOp(vals, self.fields, self.name)

    def get_launch_await_seq(self, launch_vals: Sequence[Operation | SSAValue], state: accfg.SetupOp) -> tuple[accfg.LaunchOp, accfg.AwaitOp]:
        return (
            token := accfg.LaunchOp(launch_vals, self.launch_fields, state),
            accfg.AwaitOp(token),
        )


class GemminiMvinAccelerator(GemminiOsAcceleratorBase):
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


class GemminiMvoutAccelerator(GemminiOsAcceleratorBase):
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

class GemminiExAccelerator(GemminiOsAcceleratorBase):
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
    ex = GemminiExAccelerator()
    ops_to_insert : Sequence[SSAValue | Operation] = []
    ops_to_insert.extend([
        zero := arith.Constant.from_int_and_width(0, 64),
        setup := accfg.SetupOp([zero, zero], GemminiExAccelerator.fields, "gemmini_ex"),
        *ex.get_launch_await_seq([zero,zero],setup)
    ])
    return ops_to_insert
            
