// RUN: snax-opt %s -p alloc-to-global | filecheck %s

%0 = memref.alloc() {"alignment" = 64 : i64} : memref<16x16xi8>

// CHECK:      %0 = memref.get_global @_static_const_0 : memref<16x16xi8>
// CHECK-NEXT: "memref.global"() <{sym_name = "_static_const_0", type = memref<16x16xi8>, initial_value, sym_visibility = "private"}> : () -> ()
