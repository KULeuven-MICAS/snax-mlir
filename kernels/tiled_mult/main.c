#include "data.h"
#include "mac.h"
#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

// Kernel provided via external definition
void _mlir_ciface_simple_mult(OneDMemrefI32_t *a, OneDMemrefI32_t *b,
                              OneDMemrefI32_t *d);
int main() {

  // Create memref objects for data stored in L3
  static OneDMemrefI32_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = N;
  memrefA.stride[0] = sizeof(int32_t);

  static OneDMemrefI32_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;
  memrefB.shape[0] = N;
  memrefB.stride[0] = sizeof(int32_t);

  static OneDMemrefI32_t memrefD;
  memrefD.data = &D;
  memrefD.aligned_data = memrefD.data;
  memrefD.offset = 0;
  memrefD.shape[0] = N;
  memrefD.stride[0] = sizeof(int32_t);

  (void)snrt_mcycle();
  _mlir_ciface_simple_mult(&memrefA, &memrefB, &memrefD);
  (void)snrt_mcycle();

  snrt_cluster_hw_barrier();

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < N; i++) {
    int32_t error = memrefD.aligned_data[i] - G[i];
    if (error != 0)
      nerr += 1;
  }
  return nerr;
}
