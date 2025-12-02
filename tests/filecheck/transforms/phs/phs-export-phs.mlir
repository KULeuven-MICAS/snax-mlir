// RUN: snax-opt %s -p phs-export-phs{output=\"output.mlir\"} | filecheck %s --check-prefix=STDOUT
// RUN: filecheck %s --check-prefix=FILE --input-file=output.mlir

builtin.module {
  phs.pe @acc1 with %0 (%in : i64, %in_1 : i64, %out : i64) {
    %1 = phs.choose @i_i64_i64_o_i64_0 with %0 (%in : i64, %in_1 : i64) -> i64
      0) {
        %2 = arith.addi %in, %in_1 : i64
        phs.yield %2 : i64
      }
    phs.yield %1 : i64
  }
  func.func public @addition(%arg0 : tensor<16xi64>, %arg1 : tensor<16xi64>, %arg2 : tensor<16xi64>) -> tensor<16xi64> {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<16xi64>, tensor<16xi64>) outs(%arg2 : tensor<16xi64>) attrs =  {phs_acc = @acc1} {
    ^bb0(%in : i64, %in_1 : i64, %out : i64):
      %1 = arith.addi %in, %in_1 : i64
      linalg.yield %1 : i64
    } -> tensor<16xi64>
    func.return %0 : tensor<16xi64>
  }
}

// STDOUT-NOT: phs.pe
// FILE: phs.pe
