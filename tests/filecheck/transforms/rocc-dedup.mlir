// RUN: ./compiler/snax-opt %s -p convert-acc-to-csr | filecheck %s

builtin.module {

  "acc2.accelerator"() <{
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
    %t = arith.constant 32 : i32
    %9 = "acc2.setup"(%t,%t,%t,%t,%t,%t,%t,%t,%t,%t) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 10, 0>, 
    "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1",
        "k_LOOP_WS_CONFIG_BOUNDS.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2"
        ]}> : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !acc2.state<"gemmini">


    %10 = "acc2.launch"(%t,%t,%9) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i32, i32, !acc2.state<"gemmini">) -> !acc2.token<"gemmini">
    "acc2.await"(%10) : (!acc2.token<"gemmini">) -> ()

    // An arbitrary new value
    %n = arith.constant 31 : i32
    %11 = "acc2.setup"(%n,%n,%n,%n,%9) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 4, 1>, 
        "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1", // rs1 set, but rs2 not
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",  // Both rs1 and rs2 set
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2"   // rs2 set, but rs1 not
        // strides are not set, so can be reused from the previous one.
        ]}> : (i32, i32, i32, i32, !acc2.state<"gemmini">) -> !acc2.state<"gemmini">
    %12 = "acc2.launch"(%t,%t,%11) <{
    "param_names" = ["k_LOOP_WS.rs1", "k_LOOP_WS.rs2"],
    "accelerator" = "gemmini"}> : (i32, i32, !acc2.state<"gemmini">) -> !acc2.token<"gemmini">
      "acc2.await"(%12) : (!acc2.token<"gemmini">) -> ()
    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func public @test() {
// CHECK-NEXT:     %t = arith.constant 32 : i32
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 9 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 10 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 11 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 12 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 13 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 8 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     %n = arith.constant 31 : i32
// CHECK-NEXT:     "llvm.inline_asm"(%n, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 9 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%n, %n) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 10 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %n) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 11 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     "llvm.inline_asm"(%t, %t) <{"asm_string" = ".insn r CUSTOM_3, 0x3, 8 ,x0, $0, $1", "constraints" = "r, r", "asm_dialect" = 0 : i64}> {"has_side_effects"} : (i32, i32) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
