#include "data.h"
#include "memref.h"
#include <stdio.h>
#include <stdlib.h>

void _mlir_ciface_rescale_down(OneDMemrefI8_t *output, OneDMemrefI32_t *input);

int main() {
  OneDMemrefI32_t memref_in;
  memref_in.data = &input;
  memref_in.aligned_data = memref_in.data;
  memref_in.offset = 0;
  memref_in.shape[0] = 64;
  memref_in.stride[0] = 1;

  OneDMemrefI8_t memref_out;
  memref_out.data = (int8_t *)malloc(64 * sizeof(int8_t));
  memref_out.aligned_data = memref_out.data;
  memref_out.offset = 0;
  memref_out.shape[0] = 64;
  memref_out.stride[0] = 1;

  _mlir_ciface_rescale_down(&memref_in, &memref_out);

  // Print for manual verification, acts as golden model
  for (int i = 0; i < memref_out.shape[0]; i++) {
    printf("%i: %d\n", i, memref_out.data[i]);
  }
  return 0;
}
