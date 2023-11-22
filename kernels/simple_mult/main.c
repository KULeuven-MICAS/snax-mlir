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

int8_t *allocated_pointer;

int8_t *_mlir_memref_to_llvm_alloc(int32_t size) {
  /* This calls malloc on the DMA core
   * --> requires mlir opt to compile with:
   *  --convert-memref-to-llvm="use-generic-functions index-bitwidth=32"
   * To ensure that all cores in the cluster come up with the correct
   */
  snrt_cluster_hw_barrier();
  if (snrt_is_dm_core()) {
    allocated_pointer = (int8_t *)snrt_l1alloc(size);
  }
  snrt_cluster_hw_barrier();
  return allocated_pointer;
};

int main() {
  // Allocate shared local memory
  // By avoiding allocators and bumping by a known offset a base pointer
  // (snrt_l1_next()) that is the same for all the cores in the cluster, we are
  // essentially providing the same memory regions to all the cores in this
  // cluster.

  // Allocate memory for the fields

  OneDMemrefI32_t memrefA;
  memrefA.data = (int32_t *)snrt_l1_next();
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = N;
  memrefA.stride[0] = sizeof(int32_t);

  OneDMemrefI32_t memrefB;
  memrefB.data = (int32_t *)memrefA.data + N;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = N;
  memrefB.stride[0] = sizeof(int32_t);

  OneDMemrefI32_t memrefD;
  memrefD.data = (int32_t *)memrefB.data + N;
  memrefD.aligned_data = memrefD.data;
  memrefD.offset = 0;
  memrefD.shape[0] = N;
  memrefD.stride[0] = sizeof(int32_t);

  // Copy data in shared local memory
  if (snrt_is_dm_core()) {
    snrt_dma_start_1d(memrefA.aligned_data, A,
                      (memrefA.shape[0]) * sizeof(int32_t));
    snrt_dma_start_1d(memrefB.aligned_data, B,
                      (memrefB.shape[0]) * sizeof(int32_t));
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
