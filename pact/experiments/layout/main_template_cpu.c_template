#include "stdint.h"

#include "data.h"
#include "memref.h"
#include "snax_rt.h"

/*
 * These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /sw/snRuntime/api
 * /target/snitch_cluster/sw/runtime/rtl/src
 * /target/snitch_cluster/sw/runtime/common
 * */
#include <snrt.h>

/* These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /target/snitch_cluster/sw/snax/gemm/include"
 * /target/snitch_cluster/sw/snax/mac/include"
 *
 * */
#include "snax-gemm-lib.h"
#include "snax-gemm-params.h"

uint8_t Batch = 1;
// meshRow, tileSize and meshCol are defined in snax-gemm-params.h
uint8_t M_param = M_size / meshRow;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = N_size / meshCol;
// Extracted from datagen.py in snitch_cluster repo
uint32_t rowStrideA = {rowStrideA};
uint32_t rowStrideB = {rowStrideB};
uint32_t rowStrideC = {rowStrideC};
uint32_t strideInnermostA = {strideInnermostA};
uint32_t strideInnermostB = {strideInnermostB};
uint32_t strideInnermostC = {strideInnermostC};
uint32_t ldA = {ldA};
uint32_t ldB = {ldB};
uint32_t ldC = {ldC};
uint32_t strideA = 0;
uint32_t strideB = 0;
uint32_t strideC = 0;

// Kernel provided via external definition
void _mlir_ciface_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void gemm_cpu(int Batch, int M_param, int K_param, int N_param, 
      int8_t* a_ptr, int8_t* b_ptr, int zpa, int zpb, int8_t* c_ptr, 
      int rowStrideA, int rowStrideB, int rowStrideC,
      int strideInnermostA, int strideInnermostB, int strideInnermostC, 
      int ldA, int ldB, int ldC, 
      int strideA, int strideB, int strideC) 

{{

  printf("Executing gemm_cpu with a=%p, b=%p, c=%p \n", a_ptr, b_ptr, c_ptr);

  for (int M = 0; M < M_param; M ++) {{
  for (int N = 0; N < N_param; N ++) {{
  for (int K = 0; K < K_param; K ++) {{
  for (int m = 0; m < 8; m ++) {{
  for (int n = 0; n < 8; n ++) {{
  for (int k = 0; k < 8; k ++) {{

    int8_t* addr_a = (int8_t*) ((int) a_ptr + M * ldA + K * strideInnermostA + m * rowStrideA + k);
    int8_t* addr_b = (int8_t*) ((int) b_ptr + N * ldB + K * strideInnermostB + n * rowStrideB + k);
    int32_t* addr_c = (int32_t*) ((int) c_ptr + M * ldC + N * strideInnermostC + m * rowStrideC + n * 4);

    *addr_c = *addr_c + (*addr_a * *addr_b);

  }} }} }} }} }} }}

        
}}


void _mlir_ciface_snax_qgemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
                             int32_t zpb, TwoDMemrefI32_t *c) {{

  int8_t *a_ptr = a->aligned_data;
  int8_t *b_ptr = b->aligned_data;
  int32_t *c_ptr = c->aligned_data;
  // printf("Executing snax_qgemm with a=%p, b=%p, c=%p \n", a_ptr, b_ptr, c_ptr);

  snrt_mcycle();

  // batch_gemm_cpu(Batch, M_param, K_param, N_param, a_ptr, b_ptr, 0, 0, c_ptr, strideInnermostA,
  //                strideInnermostB, strideInnermostC, ldA, ldB, ldC, strideA,
  //                strideB, strideC);

  gemm_cpu(Batch, M_param, K_param, N_param, a_ptr, b_ptr, 0, 0, c_ptr,
    rowStrideA, rowStrideB, rowStrideC, 
    strideInnermostA, strideInnermostB, strideInnermostC, 
    ldA, ldB, ldC, 
    strideA, strideB, strideC);

  snrt_mcycle();

  // printf("Finished executing snax_qgemm\n");
}}

int main() {{

  // Create memref objects for data stored in L3
  TwoDMemrefI8_t memrefA;
  memrefA.data = &A;
  memrefA.aligned_data = memrefA.data;
  memrefA.offset = 0;

  TwoDMemrefI8_t memrefB;
  memrefB.data = &B;
  memrefB.aligned_data = memrefB.data;
  memrefB.offset = 0;

  TwoDMemrefI32_t memrefC;
  memrefC.data = &C;
  memrefC.aligned_data = memrefC.data;
  memrefC.offset = 0;

  (void)snrt_mcycle();

  _mlir_ciface_matmul(&memrefA, &memrefB, &memrefC);

  snrt_cluster_hw_barrier();

  (void)snrt_mcycle();

  // Correctness check -
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int nerr = 0;
  for (int i = 0; i < M_size * N_size; i++) {{
    int32_t error = memrefC.aligned_data[i] - C_golden[i];
    if (error != 0)
      nerr += 1;
  }}

  // insert mcycle to show fault in trace
  if (nerr != 0) snrt_mcycle();

  return nerr;
}}
