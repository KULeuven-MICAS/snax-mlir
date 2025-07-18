// RUN: snax-opt %s -p alloc-to-global | filecheck %s

func.func @main() -> memref<16x16xi8> {
  %0 = memref.alloc() : memref<16x16xi8>
  func.return %0 : memref<16x16xi8>
}

// CHECK:      %0 = memref.get_global @_static_const_0 : memref<16x16xi8>
// CHECK: "memref.global"() <{sym_name = "_static_const_0", type = memref<16x16xi8>, initial_value = dense<77> : tensor<16x16xi8>, sym_visibility = "private"}> : () -> ()
