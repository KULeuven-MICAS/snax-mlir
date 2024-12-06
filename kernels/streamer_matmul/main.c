#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"
#include <snrt.h>

void _mlir_ciface_snax_main(TwoDMemrefI32_t *results);

int main() {

  TwoDMemrefI32_t results[2];

  TwoDMemrefI32_t *golden, *computed;

  golden = &results[0];
  computed = &results[1];

  // allocate zero row in tcdm
  snrt_l1alloc(256);

  (void)snrt_mcycle();
  snrt_cluster_hw_barrier();

  _mlir_ciface_snax_main(results);

  snrt_cluster_hw_barrier();
  (void)snrt_mcycle();

  // Correctness check
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int total_results = 1;
  for (int i = 0; i < 2; i++)
    total_results *= computed->shape[i];

  printf("Checking %d results...\n", total_results);

  int nerr = 0;

  for (int i = 0; i < total_results; i++) {

    if (golden->aligned_data[i] != computed->aligned_data[i]) {
      // printf("(%d) %d -> %d\n", i, golden->aligned_data[i],
      // computed->aligned_data[i]);
      nerr++;
    }
  }

  printf("Finished, nb errors: %d\n", nerr);

  if (nerr > 0)
    return 1;
  else
    return 0;
}
