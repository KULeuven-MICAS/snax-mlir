// RUN: ./compiler/snax-opt %s -p snax-lower-mcycle --print-op-generic | filecheck %s

  "snax.mcycle"() : () -> ()
//func.func @mcycle () -> () {
//  "snax.mcycle"() : () -> ()
//  func.return
//  }

// CHECK: "llvm.inline_asm"() <{asm_string = "csrr zero, mcycle", constraints = "~{memory}", asm_dialect = 0 : i64, has_side_effects}> : () -> ()

