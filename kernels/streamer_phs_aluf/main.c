#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "snrt.h"
#include <math.h> // For fabs

// Assuming OneDMemrefF64_t is defined in memref.h for doubles
void _mlir_ciface_streamer_add(OneDMemrefF64_t *A, OneDMemrefF64_t *B,
                               OneDMemrefF64_t *O);

void _mlir_ciface_debug_dart(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c,
                             int32_t when) {
  double *ptr_a, *ptr_b, *ptr_c;
  uintptr_t addr_a = (uintptr_t)_ptr_a;
  uintptr_t addr_b = (uintptr_t)_ptr_b;
  uintptr_t addr_c = (uintptr_t)_ptr_c;

  if (snrt_cluster_core_idx() == 0) {
    printf("Debugging dart op at t = %d with A at %p, B at %p, C at %p\n", when,
           ptr_a, ptr_b, ptr_c);

    for (int i = 0; i < 5; i++) {
      uint64_t val_a = *(volatile uint64_t *)(addr_a + i * 8);
      uint64_t val_b = *(volatile uint64_t *)(addr_b + i * 8);
      uint64_t val_c = *(volatile uint64_t *)(addr_c + i * 8);

      // Split into High (upper 32 bits) and Low (lower 32 bits)
      uint32_t a_hi = (uint32_t)(val_a >> 32);
      uint32_t a_lo = (uint32_t)(val_a & 0xFFFFFFFF);

      uint32_t b_hi = (uint32_t)(val_b >> 32);
      uint32_t b_lo = (uint32_t)(val_b & 0xFFFFFFFF);

      uint32_t c_hi = (uint32_t)(val_c >> 32);
      uint32_t c_lo = (uint32_t)(val_c & 0xFFFFFFFF);

      // Print using standard 32-bit hex specifiers
      printf("i%d -> A=0x%08x%08x, B=0x%08x%08x, C=0x%08x%08x\n", i, a_hi, a_lo,
             b_hi, b_lo, c_hi, c_lo);
    }
  }
}

void _mlir_ciface_debug_kernel_add(int32_t _ptr_a, int32_t _ptr_b,
                                   int32_t _ptr_c, int32_t when) {
  _mlir_ciface_debug_dart(_ptr_a, _ptr_b, _ptr_c, when);
}

int main() {
  int err = 0;

  OneDMemrefF64_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = DATA_LEN;

  OneDMemrefF64_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = DATA_LEN;

  OneDMemrefF64_t memrefO;
  memrefO.data = &O;
  memrefO.aligned_data = memrefO.data;
  memrefO.offset = 0;
  memrefO.shape[0] = DATA_LEN;

  (void)snrt_mcycle();
  _mlir_ciface_streamer_add(&memrefA, &memrefB, &memrefO);
  snrt_cluster_hw_barrier();
  (void)snrt_mcycle();

  if (snrt_cluster_core_idx() != 0)
    return 0;

  uint32_t end_csr_setup = snrt_mcycle();

  // Floating point comparison logic - BIT EXACT!
  for (uint32_t i = 0; i < DATA_LEN; i++) {
    uint64_t check_val = *(uint64_t *)((uintptr_t)&memrefO.aligned_data[i]);
    uint64_t result_val = *(uint64_t *)&G[i];
    if (check_val - result_val != 0) {
      err++;
    }
  }

  uint32_t perf_count = read_csr(0x3d0);
  printf("Accelerator Done! \n");
  printf("Accelerator Cycles: %d \n", perf_count);

  return err != 0;
}
