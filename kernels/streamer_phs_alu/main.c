#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "snrt.h"

void _mlir_ciface_streamer_add(OneDMemrefI64_t *A, OneDMemrefI64_t *B,
                               OneDMemrefI64_t *O);

void _mlir_ciface_debug_kernel_add(int32_t _ptr_a, int32_t _ptr_b,
                                   int32_t _ptr_c, int32_t when) {
  int64_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int64_t *)_ptr_a;
  ptr_b = (int64_t *)_ptr_b;
  ptr_c = (int64_t *)_ptr_c;

  if (snrt_cluster_core_idx() == 0) {
    printf("Debugging linalg op at t = %d with A at %p, B at %p, C at %p\n",
           when, ptr_a, ptr_b, ptr_c);

    for (int i = 0; i < 5; i++) {
      printf("i%d -> A=%d, B=%d, C=%d\n", i, (int32_t)ptr_a[i],
             (int32_t)ptr_b[i], (int32_t)ptr_c[i]);
    }
  }
}

void _mlir_ciface_debug_dart(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c,
                             int32_t when) {
  int64_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int64_t *)_ptr_a;
  ptr_b = (int64_t *)_ptr_b;
  ptr_c = (int64_t *)_ptr_c;

  if (snrt_cluster_core_idx() == 0) {
    printf("Debugging dart op at t = %d with A at %p, B at %p, C at %p\n", when,
           ptr_a, ptr_b, ptr_c);

    for (int i = 0; i < 5; i++) {
      printf("i%d -> A=%d, B=%d, C=%d\n", i, (int32_t)ptr_a[i],
             (int32_t)ptr_b[i], (int32_t)ptr_c[i]);
    }
  }
}

int main() {
  // Set err value for checking
  int err = 0;

  // Create memref objects for data stored in L3
  OneDMemrefI64_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = DATA_LEN;

  OneDMemrefI64_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = DATA_LEN;

  OneDMemrefI64_t memrefO;
  memrefO.data = &O;
  memrefO.aligned_data = memrefO.data;
  memrefO.offset = 0;
  memrefO.shape[0] = DATA_LEN;

  (void)snrt_mcycle();

  _mlir_ciface_streamer_add(&memrefA, &memrefB, &memrefO);

  snrt_cluster_hw_barrier();

  (void)snrt_mcycle();

  // Compare results and check if the
  // accelerator returns correct answers
  // For every incorrect answer, increment err
  if (snrt_cluster_core_idx() != 0)
    return 0;

  // Mark the end of the CSR setup cycles
  uint32_t end_csr_setup = snrt_mcycle();

  // Compare results and check if the
  // accelerator returns correct answers
  // For every incorrect answer, increment err
  uint64_t check_val;

  for (uint32_t i = 0; i < DATA_LEN; i++) {
    check_val = memrefO.aligned_data[i];
    if (check_val != G[i]) {
      err++;
    }
  }

  // Read performance counter
  uint32_t perf_count = read_csr(0x3d0);

  printf("Accelerator Done! \n");
  printf("Accelerator Cycles: %d \n", perf_count);

  return err != 0;
}
