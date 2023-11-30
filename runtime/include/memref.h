#pragma once

#include <stdint.h>

struct OneDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  uint32_t offset;
  uint32_t shape[1];
  uint32_t stride[1];
};

typedef struct OneDMemrefI32 OneDMemrefI32_t;
