#include "data.h"
#include "mac.h"
#include "memref.h"
#include "stdint.h"

#include <snrt.h>

// Kernel provided via external definition
void _mlir_ciface_simple_mult(OneDMemrefI32_t *a, OneDMemrefI32_t *b,
                              OneDMemrefI32_t *d);

void _mlir_ciface_snax_hwpe_mult(OneDMemrefI32_t *a, OneDMemrefI32_t *b,
                                 OneDMemrefI32_t *d) {
  snax_mac_setup_simple_mult(a->aligned_data, b->aligned_data, d->aligned_data,
                             a->shape[0]);
  snax_mac_launch();
  snax_mac_sw_barrier();
}

int main() {
  // Allocate shared local memory
  // By avoiding allocators and bumping by a known offset a base pointer
  // (snrt_l1_next()) that is the same for all the cores in the cluster, we are
  // essentially providing the same memory regions to all the cores in this
  // cluster.

  uint32_t constant_zero = 0;
  uint32_t constant_size = N;
  // Allocate memory for the fields

  OneDMemrefI32_t memrefA = {
      .data = (int32_t *)snrt_l1_next(),
      .aligned_data = memrefA.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  OneDMemrefI32_t memrefB = {
      .data = (int32_t *)memrefA.data + N,
      .aligned_data = memrefB.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  OneDMemrefI32_t memrefD = {
      .data = (int32_t *)memrefB.data + N,
      .aligned_data = memrefD.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  // Copy data in shared local memory
  if (snrt_is_dm_core()) {
    snrt_dma_start_1d(memrefA.aligned_data, A,
                      *(memrefA.shape[0]) * sizeof(int32_t));
    snrt_dma_start_1d(memrefB.aligned_data, B,
                      *(memrefB.shape[0]) * sizeof(int32_t));
  }

  snrt_cluster_hw_barrier();

  // Launch kernel: from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  (void)snrt_mcycle();
  _mlir_ciface_simple_mult(&memrefA, &memrefB, &memrefD);
  (void)snrt_mcycle();

  // Correctness check
  int nerr = 0;
  for (int i = 0; i < N; i++) {
    int32_t error = memrefD.aligned_data[i] - G[i];
    if (error != 0)
      nerr += 1;
  }
  return nerr;
}
