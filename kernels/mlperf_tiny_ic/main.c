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

void _mlir_ciface_run_network(FourDMemrefI8_t *output, TwoDMemrefI8_t *input);

int main() {

  {

    // Create memref objects for data stored in L3
    FourDMemrefI8_t memrefA;
    memrefA.data = &A;
    memrefA.aligned_data = memrefA.data;

    TwoDMemrefI8_t memrefB;

    // Shape and Stride need to be defined for dynamic case


    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
      printf("Running ResNet-8 \n");
    }

    snrt_cluster_hw_barrier();

    int32_t start = snrt_mcycle();

    _mlir_ciface_run_network(&memrefB, &memrefA);

    int32_t end = snrt_mcycle();


    snrt_cluster_hw_barrier();

    if (snrt_is_dm_core()) {
      printf("start = %d, end = %d, took %d \n", start, end, end - start);
    }


    snrt_cluster_hw_barrier();

    return 0;
  }
}
