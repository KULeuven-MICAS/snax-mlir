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
void _mlir_ciface_conv(FourDMemrefI32_t *o, FourDMemrefI8_t *i,
                       FourDMemrefI8_t *w);

int main() {
  {

    // Create memref objects for data stored in L3
    FourDMemrefI8_t memrefI;
    memrefI.data = &I;
    memrefI.aligned_data = memrefI.data;

    FourDMemrefI8_t memrefW;
    memrefW.data = &W;
    memrefW.aligned_data = memrefW.data;

    FourDMemrefI32_t memrefO;

    // allocate zero row in tcdm
    snrt_l1alloc(256);

    (void)snrt_mcycle();

    _mlir_ciface_conv(&memrefO, &memrefI, &memrefW);

    snrt_cluster_hw_barrier();

    (void)snrt_mcycle();

    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0)
      return 0;

    // do not check errors for now, golden model not available

    return 0;
  }
}
