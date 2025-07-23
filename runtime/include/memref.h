#pragma once

#include <stdint.h>

struct OneDMemrefI8 {
  int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
                // only used for deallocating the memref
  int8_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                        // that memref indexes
  intptr_t offset;
  intptr_t shape[1];
  intptr_t stride[1];
};

struct OneDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  intptr_t offset;
  intptr_t shape[1];
  intptr_t stride[1];
};

struct OneDMemrefI64 {
  int64_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int64_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  intptr_t offset;
  intptr_t shape[1];
  intptr_t stride[1];
};

struct TwoDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  intptr_t offset;
  intptr_t shape[2];
  intptr_t stride[2];
};

struct TwoDMemrefI8 {
  int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
                // only used for deallocating the memref
  int8_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                        // that memref indexes
  intptr_t offset;
  intptr_t shape[2];
  intptr_t stride[2];
};

struct ThreeDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  intptr_t offset;
  intptr_t shape[3];
  intptr_t stride[3];
};

struct ThreeDMemrefI8 {
  int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
                // only used for deallocating the memref
  int8_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                        // that memref indexes
  intptr_t offset;
  intptr_t shape[3];
  intptr_t stride[3];
};

struct FourDMemrefI32 {
  int32_t *data; // allocated pointer: Pointer to data buffer as allocated,
                 // only used for deallocating the memref
  int32_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                         // that memref indexes
  intptr_t offset;
  intptr_t shape[4];
  intptr_t stride[4];
};

struct FourDMemrefI8 {
  int8_t *data; // allocated pointer: Pointer to data buffer as allocated,
                // only used for deallocating the memref
  int8_t *aligned_data; // aligned pointer: Pointer to properly aligned data
                        // that memref indexes
  intptr_t offset;
  intptr_t shape[4];
  intptr_t stride[4];
};

typedef struct OneDMemrefI8 OneDMemrefI8_t;
typedef struct OneDMemrefI32 OneDMemrefI32_t;
typedef struct OneDMemrefI64 OneDMemrefI64_t;
typedef struct TwoDMemrefI8 TwoDMemrefI8_t;
typedef struct TwoDMemrefI32 TwoDMemrefI32_t;
typedef struct ThreeDMemrefI8 ThreeDMemrefI8_t;
typedef struct ThreeDMemrefI32 ThreeDMemrefI32_t;
typedef struct FourDMemrefI8 FourDMemrefI8_t;
typedef struct FourDMemrefI32 FourDMemrefI32_t;
