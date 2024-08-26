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
//#include "snax-streamer-gemm-lib.h"

#define tileSize 8
#define meshRow 8
#define meshCol 8

uint8_t Batch = 1;

/* M_param and N_param can be set to 1 for tiled versions, but not for simple
 version. 2 always works. however, it will impact performance significantly as
 computation cost doubles. For benchmarks, set to 1 */
uint8_t M_param = 2;
uint8_t K_param = K_size / tileSize;
uint8_t N_param = 2;

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

int main() {
  {

    // Create memref objects for data stored in L3
    TwoDMemrefI8_t memrefA;
    memrefA.data = &A;
    memrefA.aligned_data = memrefA.data;
    // Shape and Stride need to be defined for dynamic case
    memrefA.shape[0] = N_size;
    memrefA.shape[1] = K_size;
    memrefA.stride[0] = K_size;
    memrefA.stride[1] = 1;
    memrefA.offset = 0;

    TwoDMemrefI8_t memrefB;
    memrefB.data = &B;
    memrefB.aligned_data = memrefB.data;
    // Shape and Stride need to be defined for dynamic case
    memrefB.shape[0] = K_size;
    memrefB.shape[1] = M_size;
    memrefB.stride[0] = 1;
    memrefB.stride[1] = K_size;
    memrefB.offset = 0;


    TwoDMemrefI32_t memrefC;
    memrefC.data = &C;
    memrefC.aligned_data = memrefC.data;
    // Shape and Stride need to be defined for dynamic case
    memrefC.shape[0] = N_size;
    memrefC.shape[1] = M_size;
    memrefC.stride[0] = M_size;
    memrefC.stride[1] = 1;
    memrefC.offset = 0;

    for(int i = 0; i < 20; i++) {
      if (snrt_cluster_core_idx() == i) {
        printf("Core %d present.\n", i);
        if (snrt_is_dm_core()) {
          printf("I am a dm core\n");
        }
      }
      snrt_cluster_hw_barrier();
    }

    _mlir_ciface_streamer_matmul(&memrefA, &memrefB, &memrefC);

    snrt_cluster_hw_barrier();

    snrt_mcycle();
    int thisc = snrt_cluster_core_idx();

    for(uint8_t i = 0; i < 20; i++) {
      if (thisc == i) {
        printf("Core %d present.\n", thisc);
        if (snrt_is_dm_core()) {
          printf("I am a dm core\n");
        }
      }
      snrt_cluster_hw_barrier();
    }

    snrt_cluster_hw_barrier();


    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_hartid();
    int nerr = 0;
    if (snrt_hartid() == 0) {

      printf("Checking correctness\n");

      for (int i = 0; i < M_size * N_size; i++) {
        int32_t error = memrefC.aligned_data[i] - C_golden[i];
        if (error != 0) {
          nerr += 1;
          printf("%d) %d -> %d\n", i, (int32_t)memrefC.aligned_data[i], (int32_t)C_golden[i]);
        }
      }

    }


    return nerr;
  }
}
