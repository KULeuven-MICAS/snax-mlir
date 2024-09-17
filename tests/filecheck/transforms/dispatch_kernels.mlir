// RUN: ./compiler/snax-opt --split-input-file %s -p insert-accfg-op{accelerator=snax_alu},insert-accfg-op{accelerator=snax_gemm},dispatch-kernels --allow-unregistered-dialect --print-op-generic | filecheck %s

builtin.module {
  func.func @mnist(%arg0 : memref<?x128xi8>, %arg1 : memref<128x128xi8>, %arg2 : memref<?x128xi32>) -> memref<?x128xi32> {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %0, %1 : memref<?x128xi8>, memref<128x128xi8>, i32, i32) outs(%arg2 : memref<?x128xi32>) {
    ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = kernel.qmac %arg5, %arg6 zp_lhs : %arg3 zp_rhs : %arg4 : i32, i32, i8, i8 -> i32
      linalg.yield %2 : i32
    }
    func.return %arg2 : memref<?x128xi32>
  }
}

//CHECK: "library_call" = "snax_gemm"

// -----

builtin.module {
  func.func @mnist(%arg0 : memref<128x128xi8>, %arg1 : memref<128x128xi8>, %arg2 : memref<128x128xi32>) -> memref<128x128xi32> {
    %0 = arith.constant 1 : i32
    %1 = arith.constant 1 : i32
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1, %0, %1 : memref<128x128xi8>, memref<128x128xi8>, i32, i32) outs(%arg2 : memref<128x128xi32>) {
    ^0(%arg3 : i8, %arg4 : i8, %arg5 : i32, %arg6 : i32, %arg7 : i32):
      %2 = kernel.qmac %arg5, %arg6 zp_lhs : %arg3 zp_rhs : %arg4 : i32, i32, i8, i8 -> i32
      linalg.yield %2 : i32
    }
    func.return %arg2 : memref<128x128xi32>
  }
}

//CHECK: "library_call" = "snax_gemm_stream"

// -----

builtin.module {
  func.func public @streamer_add(%arg0 : memref<16xi64>, %arg1 : memref<16xi64>, %arg2 : memref<16xi64>) {
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<16xi64>, memref<16xi64>) outs(%arg2 : memref<16xi64>) {
    ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %0 = kernel.add %arg3, %arg4 : i64, i64 -> i64
      linalg.yield %0 : i64
    }
    func.return
  }
}

// CHECK: "library_call" = "snax_alu_stream"

// -----

builtin.module {
  func.func public @streamer_add(%arg0 : memref<?xi64>, %arg1 : memref<?xi64>, %arg2 : memref<?xi64>) {
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xi64>, memref<?xi64>) outs(%arg2 : memref<?xi64>) {
    ^0(%arg3 : i64, %arg4 : i64, %arg5 : i64):
      %0 = kernel.add %arg3, %arg4 : i64, i64 -> i64
      linalg.yield %0 : i64
    }
    func.return
  }
}

// CHECK: "library_call" = "snax_alu"
