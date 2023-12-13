#include "data.h"
#include "memref.h"
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"
#include "snax_rt.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

// Kernel provided via external definition
void _mlir_ciface_simple_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void _mlir_ciface_simple_matmul_cpu(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                    TwoDMemrefI32_t *c) {
  uint8_t Batch = 1;
  // meshRow, tileSize and meshCol are defined in snax-gemm-params.h
  uint8_t M_param = M_size / meshRow;
  uint8_t K_param = K_size / tileSize;
  uint8_t N_param = N_size / meshCol;
  int8_t *A_ptr = a->aligned_data;
  int8_t *B_ptr = b->aligned_data;
  int32_t *C_ptr = c->aligned_data;
  // Extracted from datagen.py in snitch_cluster repo
  uint32_t strideInnermostA = 256;
  uint32_t strideInnermostB = 256;
  uint32_t strideInnermostC = 256;
  uint32_t ldA = 2048;
  uint32_t ldB = 2048;
  uint32_t ldC = 1024;
  uint32_t strideA = 0;
  uint32_t strideB = 0;
  uint32_t strideC = 0;
  // delta_local_a: 64,
  // delta_local_b: 8192
  batch_gemm_cpu(Batch, M_param, K_param, N_param, A_ptr, B_ptr, C_ptr,
                 strideInnermostA, strideInnermostB, strideInnermostC, ldA, ldB,
                 ldC, strideA, strideB, strideC);
}

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
  memrefC.data = &C;
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
    int32_t error = memrefC.aligned_data[i] - C_golden[i];
    if (error != 0)
      nerr += 1;
  }
  return nerr;
}
