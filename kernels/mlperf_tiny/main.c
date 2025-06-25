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

  // struct TwoDMemrefI8 {
  //   int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
  //                 // only used for deallocating the memref
  //   int8_t *aligned_data; // aligned pointer: Pointer to properly aligned
  //   data
  //                         // that memref indexes
  //   uint32_t offset;
  //   uint32_t shape[2];
  //   uint32_t stride[2];
  // };

  // the first one is the inputs

  TwoDMemrefI8_t input;
  input.data = &data;
  input.aligned_data = input.data;
  input.offset = 0;
  input.shape[0] = 8;
  input.shape[1] = 640;
  input.stride[0] = 640;
  input.stride[1] = 1;

  int thiscore = snrt_cluster_core_idx();

  if (thiscore != 0) {
    printf("Input allocated at %p, aligned at %p, offset %d, shape [%d, %d], "
           "stride [%d, %d]\n",
           input.data, input.aligned_data, input.offset, input.shape[0],
           input.shape[1], input.stride[0], input.stride[1]);
  }
  snrt_cluster_hw_barrier();

  // the second one is the outputs
  // we don't need to initialize it, as the kernel will fill it in.
  TwoDMemrefI8_t output;

  (void)snrt_mcycle();
  snrt_cluster_hw_barrier();

  _mlir_ciface_run_network(&output, &input);

  snrt_cluster_hw_barrier();
  (void)snrt_mcycle();

  // Correctness check
  // from this point on only core 0 is required to be alive.
  if (thiscore != 0)
    return 0;

  printf("Printing the first results:\n");

  int nerr = 0;

  for (int i = 0; i < 10; i++) {

    printf("(%d): %d\n", i, output.aligned_data[i]);
  }

  printf("Finished.\n");
  return 0;
}
