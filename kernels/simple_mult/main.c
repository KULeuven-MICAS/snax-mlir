#include "data.h"
#include "mac.h"
#include "memref.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

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

// Location of matmul operands in L3 TCDM shared across cluster
int32_t *local_A;
int32_t *local_B;
int32_t *local_D;
int32_t *local_result;

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
      .data = (int32_t *)snrt_l1alloc(N * sizeof(int32_t)),
      .aligned_data = memrefA.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  OneDMemrefI32_t memrefB = {
      .data = (int32_t *)snrt_l1alloc(N * sizeof(int32_t)),
      .aligned_data = memrefB.data,
      .offset = &constant_zero,
      .shape[0] = &constant_size,
      .stride[0] = &constant_zero,
  };

  OneDMemrefI32_t memrefD = {
      .data = (int32_t *)snrt_l1alloc(N * sizeof(int32_t)),
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

  // section 1 end
  (void)snrt_mcycle();
  // section 2 start

  if (!snrt_is_dm_core())
    _mlir_ciface_simple_mult(&memrefA, &memrefB, &memrefD);

  // section 2 end
  (void)snrt_mcycle();
  // section 3 start

  snrt_cluster_hw_barrier();

  // store result back to L1

  if (snrt_is_dm_core()) {
    snrt_dma_start_1d(D, local_D, N * sizeof(int32_t));
  }

  snrt_cluster_hw_barrier();

  // section 3 end
  (void)snrt_mcycle();
  // section 4 start

  // load result from L1
  if (snrt_is_dm_core()) {
    local_result = (int32_t *)snrt_l1alloc(N * sizeof(int32_t));
    snrt_dma_start_1d(local_result, D, N * sizeof(int32_t));
  }

  snrt_cluster_hw_barrier();

  // section 4 end
  (void)snrt_mcycle();
  // section 5 start

  // do the check with compute core, dm core can return succesfully

  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  // Correctness check
  int nerr = 0;
  for (int i = 0; i < N; i++) {
    int32_t error = memrefD.aligned_data[i] - G[i];
    if (error != 0)
      nerr += 1;
  }

  // section 5 end
  (void)snrt_mcycle();

  return nerr;
}
