#include "data.h"

#include <snrt.h>

#include <stdint.h>

struct OneDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t *offset;
  uint32_t *shape[1];
  uint32_t *stride[1];
};

void _mlir_ciface_simple_mult(struct OneDMemrefI32 *A, struct OneDMemrefI32 *B,
                              struct OneDMemrefI32 *D) {
  const uint32_t n = N;
  for (uint32_t i = 0; i < n; ++i) {
    D->aligned_data[i] = A->aligned_data[i] * B->aligned_data[i];
  }
}
