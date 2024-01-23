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

void _mlir_ciface_snax_cluster_hw_barrier() {
  snrt_cluster_hw_barrier();
  return;
}

void _mlir_ciface_snax_dma_1d_transfer(size_t *source, size_t *destination,
                                       size_t size) {
  snrt_dma_start_1d((void *)destination, (void *)source, size);
  snrt_dma_wait_all();
  return;
}

void _mlir_ciface_snax_dma_2d_transfer(size_t *source, size_t *destination,
                                       size_t size, size_t src_stride,
                                       size_t dst_stride, size_t repeat) {
  // printf("Copying %d bytes from %p to %p, stridsrc %x stridedst %x rpt %d\n",
  // size, source, destination, src_stride, dst_stride, repeat);
  snrt_dma_start_2d((void *)destination, (void *)source, size, dst_stride,
                    src_stride, repeat);
}

int _mlir_ciface_snax_is_dm_core() { return snrt_is_dm_core(); }

int _mlir_ciface_snax_is_compute_core() { return snrt_is_compute_core(); }
