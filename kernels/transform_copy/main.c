#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"
#include <math.h>
#include <snrt.h>
#include <stdint.h>

void _mlir_ciface_transform_copy(TwoDMemrefI32_t *A, TwoDMemrefI32_t *B);

int main() {

  // create memref object for A
  TwoDMemrefI32_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = &A;
  memrefA.shape[0] = sqrt(N);
  memrefA.shape[1] = sqrt(N);

  // allocate memory in L1 for copy target
  TwoDMemrefI32_t memrefB;
  memrefB.data = (int32_t *)snrt_l1_next();
  memrefB.aligned_data = memrefB.data;
  memrefA.shape[0] = sqrt(N);
  memrefA.shape[1] = sqrt(N);

  snrt_cluster_hw_barrier();

  // execute copy
  if (snrt_is_dm_core()) {
    _mlir_ciface_transform_copy(&memrefA, &memrefB);
  }

  snrt_cluster_hw_barrier();

  // check if result is okay with core 0

  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;
  // Correctness check
  int nerr = 0;
  for (int i = 0; i < N; i++) {
    int32_t error = memrefB.aligned_data[i] - B[i];
    if (error != 0)
      nerr += 1;
  }

  return nerr;
}
