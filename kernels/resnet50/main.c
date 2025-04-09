#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"
#include <snrt.h>

void _mlir_ciface_snax_main(FourDMemrefI32_t *results);

int main() {

  FourDMemrefI32_t results[1];

  FourDMemrefI32_t *computed;

  computed = &results[0];

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

  printf("Finished\n");

	return 0;
}
