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

    // allocate zero row in tcdm
    snrt_l1alloc(256);

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
