#include "data.h"
#include "memref.h"
#include "stdint.h"
#include <snrt.h>

void _mlir_ciface_snax_dma_1d_transfer(OneDMemrefI32_t *a, OneDMemrefI32_t *b) {
  snrt_dma_start_1d(b->aligned_data, a->aligned_data,
                    *(a->shape[0]) * sizeof(int32_t));
  return;
}

int main() {

  uint32_t constant_zero = 0;
  uint32_t constant_size = N;

  // create memref object for A
  OneDMemrefI32_t memrefA = {
      .data = &A,
      .aligned_data = &A,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  // allocate memory in L1 for copy target
  OneDMemrefI32_t memrefB = {
      .data = (int32_t *)snrt_l1_next(),
      .aligned_data = memrefB.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  // execute copy
  if (snrt_is_dm_core()) {
    _mlir_ciface_simple_copy(&memrefA, &memrefB);
    // snrt_dma_start_1d((&memrefB)->aligned_data, (&memrefA)->aligned_data,
    // *((&memrefA)->shape[0]) * sizeof(int32_t));
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

  return nerr;
}
