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


int main() {
  {

    // Create memref objects for data stored in L3
    TwoDMemrefI8_t memrefA;
    memrefA.data = &A;
    memrefA.aligned_data = memrefA.data;
    // Shape and Stride need to be defined for dynamic case
    memrefA.shape[0] = 8;
    memrefA.shape[1] = 640;
    memrefA.stride[0] = 640;
    memrefA.stride[1] = 1;
    memrefA.offset = 0;

    (void)snrt_mcycle();

    _mlir_ciface_streamer_matmul(&memrefA, &memrefB, &memrefC);

    snrt_cluster_hw_barrier();

    (void)snrt_mcycle();

    return 0;
  }
}
