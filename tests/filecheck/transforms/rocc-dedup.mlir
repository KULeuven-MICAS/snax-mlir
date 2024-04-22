// XFAIL: *
// RUN: ./compiler/snax-opt %s -p convert-acc-to-csr | filecheck %s

builtin.module {

  "acc2.accelerator"() <{
      name            = @gemmini,
      fields = { k_LOOP_WS_CONFIG_BOUNDS.rs1=9, k_LOOP_WS_CONFIG_ADDRS_AB.rs1=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs1=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs1=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs1=13,
        k_LOOP_WS.rs1=8,
        k_LOOP_WS_CONFIG_BOUNDS.rs2=9,
        k_LOOP_WS_CONFIG_ADDRS_AB.rs2=10,
        k_LOOP_WS_CONFIG_ADDRS_DC.rs2=11,
        k_LOOP_WS_CONFIG_STRIDES_AB.rs2=12,
        k_LOOP_WS_CONFIG_STRIDES_DC.rs2=13,
        k_LOOP_WS.rs2=8
        },
      launch_addr     = 0x3c0,
      barrier         = 0x3c3
  }> : () -> ()


  func.func public @test() {
    %t = arith.constant 32 : i32
    %9 = "acc2.setup"(%t,%t,%t,%t,%t,%t,%t,%t,%t,%t,%t,%t) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 12, 0>, 
    "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs1",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs1",
        "k_LOOP_WS.rs1",
        "k_LOOP_WS_CONFIG_BOUNDS.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_AB.rs2",
        "k_LOOP_WS_CONFIG_STRIDES_DC.rs2",
        "k_LOOP_WS.rs2"
        ]}> : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !acc2.state<"gemmini">

    %10 = "acc2.launch"(%9) <{"accelerator" = "gemmini"}> : (!acc2.state<"gemmini">) -> !acc2.token<"gemmini">
      "acc2.await"(%10) : (!acc2.token<"gemmini">) -> ()

    // An arbitrary new value
    %n = arith.constant 31 : i32
    %11 = "acc2.setup"(%n,%n,%n,%n,%n,%n,%9) <{"accelerator" = "gemmini", "operandSegmentSizes" = array<i32: 6, 1>, 
        "param_names" = [ "k_LOOP_WS_CONFIG_BOUNDS.rs1", // rs1 set, but rs2 not
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs1",  // Both rs1 and rs2 set
        "k_LOOP_WS_CONFIG_ADDRS_AB.rs2",
        "k_LOOP_WS_CONFIG_ADDRS_DC.rs2",  // rs2 set, but rs1 not
        "k_LOOP_WS.rs1",                  // rs1 and rs2 set for launch-semantic op
        "k_LOOP_WS.rs2"
        ]}> : (i32, i32, i32, i32, i32, i32, !acc2.state<"gemmini">) -> !acc2.state<"gemmini">
    %12 = "acc2.launch"(%11) <{"accelerator" = "gemmini"}> : (!acc2.state<"gemmini">) -> !acc2.token<"gemmini">
      "acc2.await"(%12) : (!acc2.token<"gemmini">) -> ()
    func.return
  }
}

