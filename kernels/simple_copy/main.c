#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"
#include <snrt.h>
#include <stdint.h>

void _mlir_ciface_simple_copy(OneDMemrefI32_t *memrefA,
                              OneDMemrefI32_t *memrefB);

int main() {

  // create memref object for A
  OneDMemrefI32_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = &A;
  memrefA.offset = 0;
  memrefA.shape[0] = N;
  memrefA.stride[0] = sizeof(int32_t);

  // allocate memory in L1 for copy target
  OneDMemrefI32_t memrefB;
  memrefB.data = (int32_t *)snrt_l1_next();
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = N;
  memrefB.stride[0] = sizeof(int32_t);

  // execute copy
  if (snrt_is_dm_core()) {
    _mlir_ciface_simple_copy(&memrefA, &memrefB);
  }

  snrt_cluster_hw_barrier();

  // check if result is okay with core 0

  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  // Correctness check
  int nerr = 0;
  for (int i = 0; i < N; i++) {
    int32_t error = memrefB.aligned_data[i] - A[i];
    if (error != 0)
      nerr += 1;
  }

  if (nerr > 0)
    return 1;
  return 0;
}
