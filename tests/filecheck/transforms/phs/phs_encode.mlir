// RUN: snax-opt %s -p phs-encode --split-input-file | filecheck %s

func.func @elementwise_add_2d(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) attrs = {"phs_acc" = @acc1} {
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.addf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>

  %result2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%result, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) attrs = {"phs_acc" = @acc1}{
  ^bb0(%in0: f32, %in1: f32, %out: f32):
    %add = arith.mulf %in0, %in1 : f32
    linalg.yield %add : f32
  } -> tensor<?x?xf32>

  return %result2 : tensor<?x?xf32>
}


// CHECK: phs.pe @acc1 with %0 (%in0 : f32, %in1 : f32, %out : f32) {
// CHECK-NEXT:   %add = phs.choose @_2_opnd_0 with %0 (%in0 : f32, %in1 : f32) -> f32 {
// CHECK-NEXT:     0) arith.addf
// CHECK-NEXT:     1) arith.mulf
// CHECK-NEXT:   }
// CHECK-NEXT:   phs.yield %add : f32
// CHECK-NEXT: }

// -----

func.func @elementwise_add_2d_integer(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %init = tensor.empty(%dim0, %dim1) : tensor<?x?xi32>

  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%init : tensor<?x?xi32>) attrs = {"phs_acc" = @acc2} {
  ^bb0(%in0: i32, %in1: i32, %out: i32):
    %add = arith.addi %in0, %in1 : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>

  %result2 = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%result, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%init : tensor<?x?xi32>) attrs = {"phs_acc" = @acc2}{
  ^bb0(%in0: i32, %in1: i32, %out: i32):
    %add = arith.muli %in0, %in1 : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>

  return %result2 : tensor<?x?xi32>
}

// CHECK: phs.pe @acc2 with %0 (%in0 : i32, %in1 : i32, %out : i32) {
// CHECK-NEXT:   %add = phs.choose @_2_opnd_0 with %0 (%in0 : i32, %in1 : i32) -> i32 {
// CHECK-NEXT:     0) arith.addi
// CHECK-NEXT:     1) arith.muli
// CHECK-NEXT:   }
// CHECK-NEXT:   phs.yield %add : i32
// CHECK-NEXT: }
