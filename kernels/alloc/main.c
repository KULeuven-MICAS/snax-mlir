#include "memref.h"
#include "stdint.h"
#include <snrt.h>

// Kernel provided via external definition
// The C interface converts the pass by value to a pass by refernece!
void _mlir_ciface_simple_alloc(OneDMemrefI32_t *returned_alloc);

int8_t *allocated_pointer;

int8_t *_mlir_memref_to_llvm_alloc(uint32_t size) {
  /* This calls malloc on the DMA core
   * --> requires mlir opt to compile with:
   *  --convert-memref-to-llvm="use-generic-functions index-bitwidth=32"
   * To ensure that all cores in the cluster come up with the correct
   */
  if (snrt_is_dm_core()) {
    allocated_pointer = (int8_t *)snrt_l1alloc(size);
  }
  snrt_cluster_hw_barrier();
  return allocated_pointer;
};

int main() {
  // Allocate memory for the fields
  // static is required to have a global heap-allocated value for all cores
  static OneDMemrefI32_t memrefA;
  uint32_t core_id = snrt_cluster_core_idx();
  for (uint32_t i = 0; i < snrt_cluster_core_num(); i++) {
    // print one by one to avoid printing errors
    if (snrt_cluster_core_idx() == i) {
      printf("Core %d: Not yet allocated memrefA @ 0x%x\n", core_id,
             memrefA.data);
    }
    snrt_cluster_hw_barrier();
  }
  (void)snrt_mcycle();
  _mlir_ciface_simple_alloc(&memrefA);
  (void)snrt_mcycle();
  for (uint32_t i = 0; i < snrt_cluster_core_num(); i++) {
    // print one by one to avoid printing errors
    if (snrt_cluster_core_idx() == i) {
      printf("Core %d: Allocated a pointer to memrefA @ 0x%x\n", core_id,
             memrefA.data);
    }
    snrt_cluster_hw_barrier();
  }
  // Shape has to be 10 here for the test to work!
  if (memrefA.shape[0] != 10) {
    return 420;
  }
  uint32_t test_data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // core 0's return value is reported in CI, hence it needs to be
  // used for the final check
  if (snrt_cluster_core_idx() == 1) {
    // memcpy stack data into allocated pointer on heap
    for (size_t i = 0; i < memrefA.shape[0]; i++) {
      memrefA.data[i] = test_data[i];
    }
  }
  snrt_cluster_hw_barrier();
  // check if another core can access the same data!
  int nerr = 0;
  if (snrt_cluster_core_idx() != 1) {
    for (size_t i = 0; i < memrefA.shape[0]; i++) {
      nerr += test_data[i] - memrefA.data[i];
    }
  }
  return nerr;
}
