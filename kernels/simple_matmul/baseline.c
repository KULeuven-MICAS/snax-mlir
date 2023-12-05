// #include "data.h"
#include "memref.h"
#include "snax-gemm-lib.h"

#include <snrt.h>

#include <stdint.h>

void _mlir_ciface_simple_matmul(TwoDMemrefI8_t *A, TwoDMemrefI8_t *B,
                                TwoDMemrefI32_t *C) {
  uint8_t Batch = 1;
  uint8_t M = (uint8_t)A->shape[0];
  uint8_t K = (uint8_t)A->shape[1];
  uint8_t N = (uint8_t)B->shape[1];
  int8_t *start_addr_a = A->aligned_data;
  int8_t *start_addr_b = B->aligned_data;
  int32_t *start_addr_c = C->aligned_data;
  // TODO extract parameters below from memref?
  uint32_t strideInnermostA = 256;
  uint32_t strideInnermostB = 256;
  uint32_t strideInnermostC = 256;
  uint32_t ldA = 2048;
  uint32_t ldB = 2048;
  uint32_t ldC = 1024;
  uint32_t strideA = 0;
  uint32_t strideB = 0;
  uint32_t strideC = 0;
  batch_gemm_cpu(Batch, M, K, N, start_addr_a, start_addr_b, start_addr_c,
                 strideInnermostA, strideInnermostB, strideInnermostC, ldA, ldB,
                 ldC, strideA, strideB, strideC);
}
