// RUN: ./compiler/snax-opt -p convert-linalg-to-acc,mlir-opt[cse],acc-cse %s | filecheck %s

"builtin.module"() ({
    func.func public @simple_mult(%A: memref<?xi32>,
                                 %B: memref<?xi32>,
                                 %D: memref<?xi32>) -> () {
      linalg.generic {
          indexing_maps = [
            affine_map<(n) -> (n)>,
            affine_map<(n) -> (n)>,
            affine_map<(n) -> (n)>
          ],
          iterator_types = ["parallel"],
          library_call = "snax_hwpe_mult"
      } ins(%A, %B: memref<?xi32>, memref<?xi32>)
        outs(%D: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
        %r0 = arith.muli %a, %b : i32
        linalg.yield %r0 : i32
      }

      %i1 = "test.op"() : () -> i1

      %v_final = "scf.if"(%i1) ({
            %s = "acc2.setup"() <{accelerator = "snax_hwpe_mult", param_names = [], operandSegmentSizes = array<i32: 0, 0>}> : () -> !acc2.state<"snax_hwpe_mult">
            %v1 = "test.op"() : () -> i32

            scf.yield %v1 : i32
      }, {
            %v2 = "test.op"() : () -> i32
            %s2 = "acc2.setup"() <{accelerator = "snax_hwpe_mult", param_names = [], operandSegmentSizes = array<i32: 0, 0>}> : () -> !acc2.state<"snax_hwpe_mult">

            scf.yield %v2 : i32
      }) : (i1) -> i32

      "test.op"(%v_final) : (i32) -> ()

      linalg.generic {
          indexing_maps = [
            affine_map<(n) -> (n)>,
            affine_map<(n) -> (n)>,
            affine_map<(n) -> (n)>
          ],
          iterator_types = ["parallel"],
          library_call = "snax_hwpe_mult"
      } ins(%A, %D: memref<?xi32>, memref<?xi32>)
        outs(%B: memref<?xi32>) {
      ^bb0(%a: i32, %b: i32, %d: i32):
        %r0 = arith.muli %a, %b : i32
        linalg.yield %r0 : i32
      }

      return
    }
}): () -> ()

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   func.func public @simple_mult(%arg0 : memref<?xi32>, %arg1 : memref<?xi32>, %arg2 : memref<?xi32>) {
// CHECK-NEXT:     %0 = "memref.extract_aligned_pointer_as_index"(%arg0) : (memref<?xi32>) -> index
// CHECK-NEXT:     %1 = "memref.extract_aligned_pointer_as_index"(%arg1) : (memref<?xi32>) -> index
// CHECK-NEXT:     %2 = "memref.extract_aligned_pointer_as_index"(%arg2) : (memref<?xi32>) -> index
// CHECK-NEXT:     %3 = arith.constant 0 : index
// CHECK-NEXT:     %4 = "memref.dim"(%arg0, %3) : (memref<?xi32>, index) -> index
// CHECK-NEXT:     %5 = "acc2.setup"(%0, %1, %2, %4) <{"accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 4, 0>, "param_names" = ["A", "B", "O", "size"]}> : (index, index, index, index) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %6 = "acc2.launch"(%5) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:     "acc2.await"(%6) : (!acc2.token) -> ()
// CHECK-NEXT:     "test.op"() : () -> ()
// CHECK-NEXT:     %7 = "acc2.setup"(%2, %1, %5) <{"param_names" = ["B", "O"], "accelerator" = "snax_hwpe_mult", "operandSegmentSizes" = array<i32: 2, 1>}> : (index, index, !acc2.state<"snax_hwpe_mult">) -> !acc2.state<"snax_hwpe_mult">
// CHECK-NEXT:     %8 = "acc2.launch"(%7) <{"accelerator" = "snax_hwpe_mult"}> : (!acc2.state<"snax_hwpe_mult">) -> !acc2.token
// CHECK-NEXT:     "acc2.await"(%8) : (!acc2.token) -> ()
// CHECK-NEXT:     func.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
