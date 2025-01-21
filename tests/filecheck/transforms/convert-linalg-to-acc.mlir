// RUN: ./compiler/snax-opt -p 'convert-linalg-to-accfg,mlir-opt{executable=mlir-opt generic=true arguments=-cse,-canonicalize,-allow-unregistered-dialect,-mlir-print-op-generic,-split-input-file}' %s | filecheck %s

"builtin.module"() ({
  "accfg.accelerator"() <{
      name            = @snax_hwpe_mult,
      fields          = {A=0x3d0, B=0x3d1, O=0x3d3, vector_length=0x3d4, nr_iters=0x3d5, mode=0x3d6},
      launch_fields   = {launch=0x3c0},
      barrier         = 0x3c3
  }> : () -> ()

  func.func public @simple_mult(
    %A: memref<?xi32>,
    %B: memref<?xi32>,
    %D: memref<?xi32>
  ) -> () {
    linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
    ins(%A, %B: memref<?xi32>, memref<?xi32>)
      outs(%D: memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %d: i32):
      %r0 = arith.muli %a, %b : i32
      linalg.yield %r0 : i32
    }

    %i1 = "test.op"() : () -> i1

    %v_final = "scf.if"(%i1) ({

      linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
      ins(%A, %D: memref<?xi32>, memref<?xi32>) outs(%D: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
          %r0 = arith.muli %a, %b : i32
          linalg.yield %r0 : i32
      }
      %v1 = "test.op"() : () -> i32

      scf.yield %v1 : i32
    }, {
      %v2 = "test.op"() : () -> i32

      linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
      ins(%A, %B: memref<?xi32>, memref<?xi32>) outs(%D: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
          %r0 = arith.muli %a, %b : i32
          linalg.yield %r0 : i32
      }

      scf.yield %v2 : i32
    }) : (i1) -> i32

    "test.op"(%v_final) : (i32) -> ()

    linalg.generic { indexing_maps = [], iterator_types = ["parallel"], library_call = "snax_hwpe_mult" }
    ins(%A, %D: memref<?xi32>, memref<?xi32>) outs(%B: memref<?xi32>) {
    ^bb0(%a: i32, %b: i32, %d: i32):
      %r0 = arith.muli %a, %b : i32
      linalg.yield %r0 : i32
    }

    func.return
  }
}): () -> ()


// CHECK:       builtin.module {
// CHECK-NEXT:    "accfg.accelerator"() <{barrier = 963 : i64, fields = {A = 976 : i64, B = 977 : i64, O = 979 : i64, mode = 982 : i64, nr_iters = 981 : i64, vector_length = 980 : i64}, launch_fields = {launch = 960 : i64}, name = @snax_hwpe_mult}> : () -> ()
// CHECK-NEXT:    func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:      %0 = arith.constant 0 : i5
// CHECK-NEXT:      %1 = arith.constant 0 : index
// CHECK-NEXT:      %2 = arith.constant 1 : i32
// CHECK-NEXT:      %3 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:      %4 = arith.index_cast %3 : index to i32
// CHECK-NEXT:      %5 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:      %6 = arith.index_cast %5 : index to i32
// CHECK-NEXT:      %7 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:      %8 = arith.index_cast %7 : index to i32
// CHECK-NEXT:      %9 = "memref.dim"(%arg0, %1) : (memref<?xi32>, index) -> index
// CHECK-NEXT:      %10 = arith.index_cast %9 : index to i32
// CHECK-NEXT:      %11 = accfg.setup "snax_hwpe_mult" to ("A" = %4 : i32, "B" = %6 : i32, "O" = %8 : i32, "vector_length" = %2 : i32, "nr_iters" = %10 : i32, "mode" = %2 : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      %12 = "accfg.launch"(%0, %11) <{accelerator = "snax_hwpe_mult", param_names = ["launch"]}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:      "accfg.await"(%12) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:      %13 = "test.op"() : () -> i1
// CHECK-NEXT:      %14, %15 = scf.if %13 -> (i32, !accfg.state<"snax_hwpe_mult">) {
// CHECK-NEXT:        %16 = accfg.setup "snax_hwpe_mult" from %11 to ("A" = %4 : i32, "B" = %8 : i32, "O" = %8 : i32, "vector_length" = %2 : i32, "nr_iters" = %10 : i32, "mode" = %2 : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:        %17 = "accfg.launch"(%0, %16) <{accelerator = "snax_hwpe_mult", param_names = ["launch"]}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:        "accfg.await"(%17) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:        %18 = "test.op"() : () -> i32
// CHECK-NEXT:        scf.yield %18, %16 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %19 = "test.op"() : () -> i32
// CHECK-NEXT:        %20 = accfg.setup "snax_hwpe_mult" from %11 to ("A" = %4 : i32, "B" = %6 : i32, "O" = %8 : i32, "vector_length" = %2 : i32, "nr_iters" = %10 : i32, "mode" = %2 : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:        %21 = "accfg.launch"(%0, %20) <{accelerator = "snax_hwpe_mult", param_names = ["launch"]}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:        "accfg.await"(%21) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:        scf.yield %19, %20 : i32, !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      }
// CHECK-NEXT:      "test.op"(%14) : (i32) -> ()
// CHECK-NEXT:      %22 = accfg.setup "snax_hwpe_mult" from %15 to ("A" = %4 : i32, "B" = %8 : i32, "O" = %6 : i32, "vector_length" = %2 : i32, "nr_iters" = %10 : i32, "mode" = %2 : i32) : !accfg.state<"snax_hwpe_mult">
// CHECK-NEXT:      %23 = "accfg.launch"(%0, %22) <{accelerator = "snax_hwpe_mult", param_names = ["launch"]}> : (i5, !accfg.state<"snax_hwpe_mult">) -> !accfg.token<"snax_hwpe_mult">
// CHECK-NEXT:      "accfg.await"(%23) : (!accfg.token<"snax_hwpe_mult">) -> ()
// CHECK-NEXT:      func.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
