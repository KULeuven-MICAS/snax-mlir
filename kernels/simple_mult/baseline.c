#include "data.h"
#include "memref.h"

#include <snrt.h>

#include <stdint.h>

void _mlir_ciface_simple_mult(OneDMemrefI32_t *A, OneDMemrefI32_t *B,
                              OneDMemrefI32_t *D) {
  const uint32_t n = N;
  for (uint32_t i = 0; i < n; ++i) {
    D->aligned_data[i] = A->aligned_data[i] * B->aligned_data[i];
  }
}
