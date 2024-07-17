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
 * /target/snitch_cluster/sw/snax/streamer-gemm/include"
 * /target/snitch_cluster/sw/snax/mac/include"
 *
 * */
#include "snax-streamer-gemm-lib.h"

#define tileSize 8
#define meshRow 8
#define meshCol 8

uint8_t Batch = 1;
// meshRow, tileSize and meshCol are defined in snax-gemm-params.h
uint8_t M_param = M_size / meshRow;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = N_size / meshCol;
// Extracted from datagen.py in snitch_cluster repo
uint32_t strideInnermostA = 256;
uint32_t strideInnermostB = 256;
uint32_t strideInnermostC = 256;
uint32_t ldA = 512;
uint32_t ldB = 512;
uint32_t ldC = 512;
uint32_t strideA = 0;
uint32_t strideB = 0;
uint32_t strideC = 0;

// Kernel provided via external definition
void _mlir_ciface_streamer_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                  TwoDMemrefI32_t *c);

void _mlir_ciface_snax_gemm(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b, int32_t zpa,
                            int32_t zpb, TwoDMemrefI32_t *c) {
  {

    int local_delta_a = (int)a->aligned_data - (int)snrt_l1_next();
    int local_delta_b = (int)b->aligned_data - (int)snrt_l1_next();
    int local_delta_c = (int)c->aligned_data - (int)snrt_l1_next();
    printf("Executing snax_gemm with a=%p, b=%p, c=%p \n",
           (int8_t)a->aligned_data, (int8_t)b->aligned_data,
           (int32_t)c->aligned_data);

    set_streamer_csr(K_param, N_param, M_param, strideInnermostA, ldA, 8,
                     strideInnermostB, ldB, 8, strideInnermostC, ldC, 32,
                     local_delta_a, local_delta_b, local_delta_c);
    set_streamer_start();
    set_block_gemm_csr(K_param, N_param, M_param, 0);

    snrt_mcycle();

    set_block_gemm_start();

    printf("Waiting for snax_gemm\n");

    wait_streamer_gemm();

    snrt_mcycle();

    printf("Finished executing snax_gemm\n");
  }
}

int main() {
  {

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

    _mlir_ciface_streamer_matmul(&memrefA, &memrefB, &memrefC);

    snrt_cluster_hw_barrier();

    (void)snrt_mcycle();

    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0)
      return 0;

    int nerr = 0;
    for (int i = 0; i < M_size * N_size; i++) {
      {
        int32_t error = memrefC.aligned_data[i] - C_golden[i];
        // printf("%d) %d -> %d\n", i, (int32_t)memrefC.aligned_data[i],
        //        (int32_t)C_golden[i]);
        if (error != 0)
          nerr += 1;
      }
    }

    // insert mcycle to show fault in trace
    if (nerr != 0)
      snrt_mcycle();

    return nerr;
  }
}
