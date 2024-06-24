#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "snrt.h"

void _mlir_ciface_streamer_add(OneDMemrefI32_t *A, OneDMemrefI32_t *B,
                               OneDMemrefI32_t *O);

int main() {
  // Set err value for checking
  int err = 0;

  // Create memref objects for data stored in L3
  OneDMemrefI32_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = DATA_LEN;

  OneDMemrefI32_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = DATA_LEN;

  OneDMemrefI32_t memrefO;
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

    // memrefO is int32 type, but data is i64
    check_val = memrefO.aligned_data[i * 2];
    if (check_val != G[i]) {
      err++;
    }
  }

  // Read performance counter
  uint32_t perf_count = read_csr(0x3d0);

  printf("Accelerator Done! \n");
  printf("Accelerator Cycles: %d \n", perf_count);
  printf("Number of errors: %d \n", err);

  return err;
}
