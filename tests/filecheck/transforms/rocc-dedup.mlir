// RUN: snax-opt %s -p convert-accfg-to-csr | filecheck %s

builtin.module {

  "accfg.accelerator"() <{
      name            = @gemmini,
      fields = { k_LOOP_WS_CONFIG_BOUNDS.rs1=9, k_LOOP_WS_CONFIG_ADDRS_AB.rs1=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs1=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs1=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs1=13,
        k_LOOP_WS_CONFIG_BOUNDS.rs2=9,
        k_LOOP_WS_CONFIG_ADDRS_AB.rs2=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs2=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs2=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs2=13
        },
      launch_fields   = {
        k_LOOP_WS.rs1=8,
        k_LOOP_WS.rs2=8},
      barrier         = 0x0BAD
  }> : () -> ()


  func.func public @test() {
    %t = arith.constant 32 : i64
    %9 = "accfg.setup"(%t,%t,%t,%t,%t,%t,%t,%t,%t) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 9, 0>,
    "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1",
        "k_LOOP_WS_CONFIG_BOUNDS.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2"
        ]}> : (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> !accfg.state<"gemmini">


    %10 = "accfg.launch"(%t,%t,%9) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i64, i64, !accfg.state<"gemmini">) -> !accfg.token<"gemmini">
    "accfg.await"(%10) : (!accfg.token<"gemmini">) -> ()

    // An arbitrary new value
    %n = arith.constant 31 : i64
    %11 = "accfg.setup"(%n,%n,%n,%n,%n,%9) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 5, 1>,
        "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1", // rs1 set, but rs2 not
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",  // Both rs1 and rs2 set
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",  // rs2 set, but rs1 not
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2"   // rs2 set, but was never set before in this chain because of deduplication hoisting
        // strides are not set, so can be reused from the previous one.
        ]}> : (i64, i64, i64, i64, i64, !accfg.state<"gemmini">) -> !accfg.state<"gemmini">
    %12 = "accfg.launch"(%t,%t,%11) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i64, i64, !accfg.state<"gemmini">) -> !accfg.token<"gemmini">
      "accfg.await"(%12) : (!accfg.token<"gemmini">) -> ()
    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @test() {
// CHECK-NEXT:     %t = arith.constant 32 : i64
// CHECK-NEXT:     %0 = arith.constant 0 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 9 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 10 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 11 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %0) <{asm_string = ".insn r CUSTOM_3, 0x3, 12 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 13 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 8 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     %n = arith.constant 31 : i64
// CHECK-NEXT:     "llvm.inline_asm"(%n, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 9 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%n, %n) <{asm_string = ".insn r CUSTOM_3, 0x3, 10 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %n) <{asm_string = ".insn r CUSTOM_3, 0x3, 11 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %n) <{asm_string = ".insn r CUSTOM_3, 0x3, 12 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{asm_string = ".insn r CUSTOM_3, 0x3, 8 ,x0, $0, $1", constraints = "r, r", asm_dialect = 0 : i64, has_side_effects}> : (i64, i64) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
