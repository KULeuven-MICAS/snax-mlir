// RUN: ./compiler/snax-opt %s -p snax-lower-mcycle --print-op-generic | filecheck %s
func.func @mcycle () -> () {
  "snax.mcycle"() : () -> ()
  func.return
  }

// CHECK: %0 = "llvm.inline_asm"() <{"asm_string" = "csrr $0, mcycle", "constraints" = "=r,~{memory}", "asm_dialect" = 0 : i64}> {"has_side_effects"} : () -> i32

