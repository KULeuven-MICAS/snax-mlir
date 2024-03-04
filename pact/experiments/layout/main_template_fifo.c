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
uint32_t strideInnermostA = {strideInnermostA};
uint32_t strideInnermostB = {strideInnermostB};
uint32_t strideInnermostC = {strideInnermostC};
uint32_t ldA = {ldA};
uint32_t ldB = {ldB};
uint32_t ldC = {ldC};
uint32_t strideA = 0;
uint32_t strideB = 0;
uint32_t strideC = 0;


// Set STREAMER configuration CSR
void set_streamer_csr(int8_t* a_ptr, int8_t* b_ptr, int32_t* c_ptr) {{
    // loop bounds, from innermost to outermost
    write_csr(960, 2);
    write_csr(961, 2);
    write_csr(962, 2);

    // temporal strides for A
    write_csr(963, {strideInnermostA});
    write_csr(964, 0);
    write_csr(965, {ldA});

    // temporal strides for B
    write_csr(966, {strideInnermostB});
    write_csr(967, {ldB});
    write_csr(968, 0);

    // temporal strides for C
    write_csr(969, 0);
    write_csr(970, {strideInnermostC});
    write_csr(971, {ldC});

    // spatial strides for A
    write_csr(972, 1);
    write_csr(973, 8);

    // spatial strides for B
    write_csr(974, 1);
    write_csr(975, 8);

    // spatial strides for C
    write_csr(976, 4);
    write_csr(977, 32);

    // base ptr for A
    write_csr(978, (uint32_t)(a_ptr));

    // base ptr for B
    write_csr(979, (uint32_t)(b_ptr));

    // base ptr for C
    write_csr(980, (uint32_t)(c_ptr));
}}

// Set CSR to start STREAMER
void set_streamer_start() {{ write_csr(981, 1); }}

// Set GEMM configuration CSR
void set_block_gemm_csr() {{
    // set loop bounds, from M to K to N
    write_csr(982, 2);
    write_csr(983, 2);
    write_csr(984, 2);

    // set subtraction a and b
    write_csr(985, 0);
}}

// Set CSR to start GEMM
void set_block_gemm_start() {{ write_csr(986, 1); }}

// Poll until Streamer and GEMM accelerator finish
void wait_streamer_gemm() {{
    write_csr(981, 1);
    write_csr(986, 1);
}}

// Kernel provided via external definition
void _mlir_ciface_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                TwoDMemrefI32_t *c);

void _mlir_ciface_snax_qgemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
                             int32_t zpb, TwoDMemrefI32_t *c) {{

    int8_t *a_ptr = a->aligned_data;
    int8_t *b_ptr = b->aligned_data;
    int32_t *c_ptr = c->aligned_data;
    printf("Executing snax_qgemm with a=%p, b=%p, c=%p \n", a_ptr, b_ptr, c_ptr);

    set_streamer_csr(a_ptr, b_ptr, c_ptr);
    set_streamer_start();
    set_block_gemm_csr();
    set_block_gemm_start();
    wait_streamer_gemm();


    printf("Finished executing snax_qgemm\n");
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
  return nerr;
}}
