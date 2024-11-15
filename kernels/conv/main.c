#include "stdint.h"

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
void _mlir_ciface_conv(int32_t *test);

int main() {
  {


    FourDMemrefI32_t results[2];

    FourDMemrefI32_t *golden, *computed;
    golden = &results[0];
    computed = &results[1];

    int32_t* some_data = results;

    // allocate zero row in tcdm
    snrt_l1alloc(256);

    (void)snrt_mcycle();
    snrt_cluster_hw_barrier();

    _mlir_ciface_conv(results);

    snrt_cluster_hw_barrier();
    (void)snrt_mcycle();

    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0)
      return 0;

    printf("Got golden result:\n");
    printf("Pointer at %p\n", golden->data);
    printf("Aligned Pointer at %p\n", golden->aligned_data);


    printf("Got computed result:\n");
    printf("Pointer at %p\n", computed->data);
    printf("Aligned Pointer at %p\n", computed->aligned_data);

    int total_results = 1;
    for (int i = 0; i < 4; i++) total_results *= computed->shape[i];

    printf("Checking %d results...\n", total_results);

    int nerr = 0;

    for(int i = 0; i < total_results; i ++) {

      if (golden->aligned_data[i] != computed->aligned_data[i]) {
        printf("(%d) %d -> %d\n", i, golden->aligned_data[i], computed->aligned_data[i]);
        nerr++;
      }
    }

    printf("Finished, nb errors: %d\n", nerr);

    if (nerr > 0) return 1;
    else return 0;

  }
}
