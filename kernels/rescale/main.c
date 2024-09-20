#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "snrt.h"

void _mlir_ciface_rescale(OneDMemrefI64_t *A, OneDMemrefI64_t *O);

int main() {
  // Set err value for checking
  int err = 0;

  // Create memref objects for data stored in L3
  OneDMemrefI64_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = DATA_LEN;

  OneDMemrefI64_t memrefO;
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

 return err;
}
