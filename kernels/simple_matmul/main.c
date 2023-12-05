#include "data.h"
#include "memref.h"
#include "snax-gemm-lib.h"
#include "snax_rt.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

// Kernel provided via external definition
void _mlir_ciface_simple_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

// void _mlir_ciface_snax_hwpe_mult(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
//                                 TwoDMemrefI32_t *c) {
//
//   set_batch_gemm(a->aligned_data, b->aligned_data, c->aligned_data,
//                              a->shape[0]);
//   start_batch_gemm();
//   wait_batch_gemm();
// }

int main() {

  // Create memref objects for data stored in L1
  TwoDMemrefI8_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;
  memrefA.shape[0] = M_size;
  memrefA.shape[1] = K_size;
  memrefA.stride[0] = sizeof(int8_t);
  memrefA.stride[1] = sizeof(int8_t);

  TwoDMemrefI8_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefA.offset = 0;
  memrefA.shape[0] = K_size;
  memrefA.shape[1] = N_size;
  memrefA.stride[0] = sizeof(int8_t);
  memrefA.stride[1] = sizeof(int8_t);

  TwoDMemrefI32_t memrefC;
  memrefC.data = &G;
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;
  memrefC.shape[0] = M_size;
  memrefC.shape[1] = N_size;
  memrefC.stride[0] = sizeof(int32_t);
  memrefC.stride[1] = sizeof(int32_t);

  (void)snrt_mcycle();
  _mlir_ciface_simple_matmul(&memrefA, &memrefB, &memrefC);
  (void)snrt_mcycle();

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {
    int32_t error = memrefC.aligned_data[i] - G[i];
    if (error != 0)
      nerr += 1;
  }
  return nerr;
}
