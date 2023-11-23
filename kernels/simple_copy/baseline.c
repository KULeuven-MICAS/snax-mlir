#include "memref.h"
#include <snrt.h>
#include <stdint.h>

void _mlir_ciface_simple_copy(OneDMemrefI32_t *memrefA,
                              OneDMemrefI32_t *memrefB) {
  snrt_dma_start_1d(memrefB->aligned_data, memrefA->aligned_data,
                    *(memrefA->shape[0]) * sizeof(int32_t));
}
