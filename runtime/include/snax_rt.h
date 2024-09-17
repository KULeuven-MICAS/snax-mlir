#pragma once

#include <snrt.h>
#include <stdint.h>

int8_t *allocated_pointer;

int8_t *_mlir_memref_to_llvm_alloc(uint32_t size) {
  /* This calls malloc on the DMA core
   * --> requires mlir opt to compile with:
   *  --convert-memref-to-llvm="use-generic-functions index-bitwidth=32"
   * To ensure that all cores in the cluster come up with the correct
   */
  if (snrt_is_dm_core()) {
    allocated_pointer = (int8_t *)snrt_l1alloc(size);
  }
  snrt_cluster_hw_barrier();
  return allocated_pointer;
};

typedef struct alloc_result {
  void *pointer;
  void *aligned_pointer;
} alloc_result_t;

alloc_result_t *allocated_result;

alloc_result_t *_mlir_ciface_snax_alloc_l1(uint32_t size, uint32_t alignment) {

  if (snrt_is_dm_core()) {
    // printf("Allocating %d bytes with alignment %d\n", size, alignment);

    void *next_ptr = snrt_l1_next();
    // calculate extra size needed to allocate for correct alignment
    uint32_t extra_size = alignment - ((int32_t)next_ptr % alignment);
    void *allocated_pointer = snrt_l1alloc(size + extra_size);
    void *aligned_pointer = (void *)((int32_t)allocated_pointer + extra_size);

    allocated_result->pointer = allocated_pointer;
    allocated_result->aligned_pointer = aligned_pointer;
  }

  snrt_cluster_hw_barrier();
  return allocated_result;
}

void _mlir_ciface_snax_dump_l1() {
  snrt_alloc_init();
  // keep first row of the tcdm free for zero wizardry
  // this has a similar function to the zero register of risc-v
  snrt_l1alloc(256);
  // memset all to 0 with DMA
  if (snrt_is_dm_core()) {
    snrt_dma_start_2d((int32_t*) 0x10000040, (int32_t*) 0x10000000, 0x40, 0x40, 0, 0x1800);
    snrt_dma_wait_all();
  }
  snrt_cluster_hw_barrier();
  return;
}

void _mlir_ciface_snax_cluster_hw_barrier() {
  snrt_cluster_hw_barrier();
  return;
}

void _mlir_ciface_snax_dma_1d_transfer(size_t *source, size_t *destination,
                                       size_t size) {
  // printf("Copying %d bytes from %p to %p\n", size, (void *)source,
  //        (void *)destination);
  snrt_dma_start_1d((void *)destination, (void *)source, size);
  snrt_dma_wait_all();
  return;
}

void _mlir_ciface_snax_dma_2d_transfer(size_t *source, size_t *destination,
                                       size_t size, size_t src_stride,
                                       size_t dst_stride, size_t repeat) {
  // printf("Copying %d bytes from %p to %p, stridsrc %x stridedst %x rpt %d\n",
  //        size, source, destination, src_stride, dst_stride, repeat);
  snrt_dma_start_2d((void *)destination, (void *)source, size, dst_stride,
                    src_stride, repeat);
  snrt_dma_wait_all();
}

uint32_t _mlir_ciface_snax_cluster_core_idx() {
  return snrt_cluster_core_idx();
}
