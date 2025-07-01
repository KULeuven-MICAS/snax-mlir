#include "data.h"
#include "memref.h"
#include "snax_rt.h"
#include "stdint.h"
#include <snrt.h>

void _mlir_ciface_run_network(TwoDMemrefI8_t *input, TwoDMemrefI8_t *output);

void _mlir_ciface_debug_dart(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c,
                             int32_t when) {
  int8_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int8_t *)_ptr_a;
  ptr_b = (int8_t *)_ptr_b;
  ptr_c = (int8_t *)_ptr_c;

  if (snrt_cluster_core_idx() == 0) {
    printf("Debugging dart op at t = %d with A at %p, B at %p, C at %p\n", when,
           ptr_a, ptr_b, ptr_c);

    for (int i = 0; i < 5; i++) {
      printf("i%d -> A=%d, B=%d, C=%d\n", i, (int32_t)ptr_a[i],
             (int32_t)ptr_b[i], (int32_t)ptr_c[i]);
    }
  }
}

int main() {

  // run_network takes in two memref structs:

  // the first one is the inputs

  TwoDMemrefI8_t input_memref;
  input_memref.data = &input;
  input_memref.aligned_data = input_memref.data;
  input_memref.offset = 0;
  input_memref.shape[0] = 8;
  input_memref.shape[1] = 640;
  input_memref.stride[0] = 640;
  input_memref.stride[1] = 1;

  snrt_cluster_hw_barrier();

  // the second one is the outputs
  // we don't need to initialize it, as the kernel will fill it in.
  TwoDMemrefI8_t output_memref;

  (void)snrt_mcycle();
  snrt_cluster_hw_barrier();

  _mlir_ciface_run_network(&output_memref, &input_memref);

  snrt_cluster_hw_barrier();
  (void)snrt_mcycle();

  // Correctness check
  // from this point on only core 0 is required to be alive.
  int thiscore = snrt_cluster_core_idx();
  if (thiscore != 0)
    return 0;

  int se = 0;

  for (int i = 0; i < output_memref.shape[1]; i++) {
    int err = (output_memref.aligned_data[i] - output[i]);
    se += (err) * (err);
  }

  int mse = se / output_memref.shape[1];

  printf("MSE: %d\n", mse);

  if (mse > 5) {
    printf("Error: MSE is too high (%d)\n", mse);
    return 1;
  }

  printf("Finished.\n");
  return 0;
}
