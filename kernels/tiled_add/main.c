#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

// Kernel provided via external definition
void _mlir_ciface_streamer_add_tiled(OneDMemrefI64_t *a, OneDMemrefI64_t *b,
                                     OneDMemrefI64_t *d);
int main() {

  // Create memref objects for data stored in L3
  static OneDMemrefI64_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = DATA_LEN;
  memrefA.stride[0] = sizeof(int64_t);

  static OneDMemrefI64_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = DATA_LEN;
  memrefB.stride[0] = sizeof(int64_t);

  static OneDMemrefI64_t memrefO;
  memrefO.data = &O;
  memrefO.aligned_data = memrefO.data;
  memrefO.offset = 0;
  memrefO.shape[0] = DATA_LEN;
  memrefO.stride[0] = sizeof(int64_t);

  (void)snrt_mcycle();
  _mlir_ciface_streamer_add_tiled(&memrefA, &memrefB, &memrefO);
  (void)snrt_mcycle();

  snrt_cluster_hw_barrier();

#ifdef NO_CHECK
  // No correctness check =
  // Always finish as if nothing happened
  return 0;
#else
  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < DATA_LEN; i++) {
    int64_t error = memrefO.aligned_data[i] - G[i];
    if (error != 0)
      // printf("%d) %d -> %d\n",i, (int32_t)memrefO.aligned_data[i],
      // (int32_t)G[i]);
      nerr += 1;
  }

  if (nerr > 0)
    ;
  return 1;
  return 0;
#endif
}
