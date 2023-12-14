#include "data.h"
#include "memref.h"
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"
#include "snax_rt.h"
#include "stdint.h"

#include <snrt.h>
#include <stdint.h>

uint8_t Batch = 1;
// meshRow, tileSize and meshCol are defined in snax-gemm-params.h
uint8_t M_param = M_size / meshRow;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = N_size / meshCol;
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

// Kernel provided via external definition
void _mlir_ciface_simple_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void _mlir_ciface_simple_matmul_cpu(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                    TwoDMemrefI32_t *c) {
  int8_t *A_ptr = a->aligned_data;
  int8_t *B_ptr = b->aligned_data;
  int32_t *C_ptr = c->aligned_data;
  batch_gemm_cpu(Batch, M_param, K_param, N_param, A_ptr, B_ptr, C_ptr,
                 strideInnermostA, strideInnermostB, strideInnermostC, ldA, ldB,
                 ldC, strideA, strideB, strideC);
}

int main() {
  // Allocate space in TCDM
  // We put the data in different banks, but we don't interleave the data for
  // now.
  //
  //  | A | x | x | x |  --> A in banks 0 - 7  --> (8/32 banks used)*
  //                                               (int8 --> 8 elements/bank)
  //                                               1 row --> 64 elements
  //  | x | B | x | x |  --> B in banks 7 - 15 --> (8/32 banks used)*
  //                                               (8 elements/bank)*32 banks
  //                                               1 row --> 64 elements
  //  | C | C | C | C |  --> C in banks 0 - 31 --> (32/32 banks used)*
  //                                               (2 elements/bank)* 32 bank
  //                                               1 row --> 64 elements
  //  | x | x | x | x |
  //
  //  32 banks -->  1 row = 32 banks * 8 bytes --> 256 adresses further

  static int8_t *allocated_a;
  static int8_t *allocated_b;
  static int32_t *allocated_c;

  // Transfer data from L3 to L1
  // Using DMA only
  if (snrt_is_dm_core()) {
    // calculation in bytes directly
    allocated_a = (int8_t *)snrt_l1alloc(256 * M_size * K_size / 64);
    allocated_b = (int8_t *)snrt_l1alloc(256 * K_size * N_size / 64);
    allocated_c = (int32_t *)snrt_l1alloc(256 * M_size * K_size / 64);
  }

  // Create memref descriptors for data stored in L1
  TwoDMemrefI8_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.shape[0] = M_size;
  memrefA.shape[1] = K_size;
  // These are not considered correctly right now
  memrefA.offset = 0;
  memrefA.stride[0] = sizeof(int8_t);
  memrefA.stride[1] = sizeof(int8_t);

  TwoDMemrefI8_t memrefB;
  memrefB.data = &B;
  // Data is stored in banks 8 - 15, so increment by 8banks*8bytes = 64
  memrefB.aligned_data = memrefB.data + 64;
  memrefB.shape[0] = K_size;
  memrefB.shape[1] = N_size;
  // These are not considered correctly right now
  memrefB.offset = 0;
  memrefB.stride[0] = sizeof(int8_t);
  memrefB.stride[1] = sizeof(int8_t);

  TwoDMemrefI32_t memrefC;
  memrefC.data = &C;
  memrefC.aligned_data = memrefC.data;
  memrefC.shape[0] = M_size;
  memrefC.shape[1] = N_size;
  // These are not considered correctly right now
  memrefC.offset = 0;
  memrefC.stride[0] = sizeof(int32_t);
  memrefC.stride[1] = sizeof(int32_t);
  if (snrt_is_dm_core()) {
    load_input_data(Batch, M_size / meshRow, K_size / tileSize,
                    N_size / meshCol, memrefA.aligned_data,
                    memrefB.aligned_data, A, B, strideInnermostA,
                    strideInnermostB, ldA, ldB, strideA, strideB);
  }

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
