// RUN: snax-opt --split-input-file -p test-add-mcycle-around-loop %s | filecheck %s

builtin.module {
  func.func @add_mcycles_to_me() {
    %t = "test.op"() : () -> index
    scf.for %arg3 = %t to %t step %t {
      scf.for %arg4 = %t to %t step %t {
        "test.op"() : () -> ()
      }
    }
    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @add_mcycles_to_me() {
// CHECK-NEXT:     %t = "test.op"() : () -> index
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     scf.for %arg3 = %t to %t step %t {
// CHECK-NEXT:       scf.for %arg4 = %t to %t step %t {
// CHECK-NEXT:         "test.op"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// -----

builtin.module {
  func.func @add_mcycles_to_me_twice() {
    %t = "test.op"() : () -> index
    scf.for %arg3 = %t to %t step %t {
      scf.for %arg4 = %t to %t step %t {
        "test.op"() : () -> ()
      }
    }
    scf.for %arg1 = %t to %t step %t {
      "test.op"() : () -> ()
    }
    func.return
  }
}

// CHECK: builtin.module {
// CHECK-NEXT:   func.func @add_mcycles_to_me_twice() {
// CHECK-NEXT:     %t = "test.op"() : () -> index
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     scf.for %arg3 = %t to %t step %t {
// CHECK-NEXT:       scf.for %arg4 = %t to %t step %t {
// CHECK-NEXT:         "test.op"() : () -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     scf.for %arg1 = %t to %t step %t {
// CHECK-NEXT:       "test.op"() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     "snax.mcycle"() : () -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
