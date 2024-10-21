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

// Kernel provided via external definition
void _mlir_ciface_streamer_matmul(TwoDMemrefI8_t *a, TwoDMemrefI8_t *b,
                                  TwoDMemrefI32_t *d);

int main() {
  {

    // Create memref objects for data stored in L3
    TwoDMemrefI8_t memrefA;
    memrefA.data = &A;
    memrefA.aligned_data = memrefA.data;
    memrefA.shape[0] = M_size;
    memrefA.shape[1] = K_size;
    memrefA.stride[0] = K_size;
    memrefA.stride[1] = 1;
    memrefA.offset = 0;

    TwoDMemrefI8_t memrefB;
    memrefB.data = &B;
    memrefB.aligned_data = memrefB.data;
    memrefB.shape[0] = N_size;
    memrefB.shape[1] = M_size;
    memrefB.stride[0] = 1;
    memrefB.stride[1] = K_size;
    memrefB.offset = 0;

    TwoDMemrefI32_t memrefD;
    memrefD.data = &D;
    memrefD.aligned_data = memrefD.data;
    memrefD.shape[0] = M_size;
    memrefD.shape[1] = N_size;
    memrefD.stride[0] = 1;
    memrefD.stride[1] = K_size;
    memrefD.offset = 0;

    snrt_cluster_hw_barrier();

    snrt_mcycle();

    _mlir_ciface_streamer_matmul(&memrefA, &memrefB, &memrefD);

    snrt_mcycle();

    snrt_cluster_hw_barrier();

    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0)
      return 0;

#ifdef NO_CHECK
    // No correctness check =
    // Always finish as if nothing happened
    return 0;
#endif
    int nerr = 0;

    for (int i = 0; i < M_size * N_size; i++) {
      {
        int32_t error = memrefD.aligned_data[i] - D_golden[i];
        // printf("%d) %d -> %d\n", i, (int32_t)memrefD.aligned_data[i],
        //        (int32_t)D[i]);
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
