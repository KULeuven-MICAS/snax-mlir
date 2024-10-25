#include "stdint.h"

#include "data.h"
#include "memref.h"
#include "snax_rt.h"

/*
 * These libraries are included from github.com/KULeuven-MICAS/snitch_cluster
 * Interested users, might want to look at:
 *
 * /sw/snRuntime/api
 * /target/snitch_cluster/sw/runtime/rtl/src
 * /target/snitch_cluster/sw/runtime/common
 * */
#include <snrt.h>
#include "snax-xdma-lib.h"

// Kernel provided via external definition
void _mlir_ciface_toy(TwoDMemrefI32_t *o, FourDMemrefI8_t *i, FourDMemrefI8_t *w, TwoDMemrefI8_t *b);

void _mlir_ciface_snax_xdma(FourDMemrefI8_t* input, TwoDMemrefI8_t* not_used, FourDMemrefI8_t* output) {

  int8_t *a_ptr = input->aligned_data;
  int8_t *b_ptr = output->aligned_data;


  //printf("Executing xdma with a=%p, b=%p\n", a_ptr, b_ptr);

  // There are three extensions in xdma
  // VerilogMemset, Maxpool, Transposer
  // 0            , 1      , 2
  // We want to only use Maxpool
  // Hence we need to disable the 0 and 2
  // and we set the maxpool csr to 16 since we need 4x4 pooling
  if (xdma_disable_dst_ext(0) != 0) {
      printf("Error in disabling xdma extension 0\n");
  } else {
      //printf("The xdma extension 0 is disabled\n");
  }

  uint32_t ext_param_maxpool_size[1] = {256};
  if (xdma_enable_dst_ext(1, ext_param_maxpool_size) != 0) {
      printf("Error in enabling xdma extension 1\n");
  } else {
      //printf("The xdma extension 1 is enabled\n");
  }

  if (xdma_disable_dst_ext(2) != 0) {
      printf("Error in disabling xdma extension 2\n");
  } else {
      //printf("The xdma extension 2 is disabled\n");
  }

  // --------------------- Configure the AGU --------------------- //
  uint32_t sstride_src[1] = {8};
  uint32_t sstride_dst[1] = {8};
  uint32_t tstride_src[3] = {8, 128, 2048};
  uint32_t tbound_src[3] = {16, 16, 2};
  uint32_t tstride_dst[1] = {8};
  uint32_t tbound_dst[1] = {2};

  int task_id;
  if (xdma_memcpy_nd( a_ptr, b_ptr,
                      sstride_src, sstride_dst, 3, tstride_src,
                      tbound_src, 1, tstride_dst, tbound_dst, 0xFFFFFFFF,
                      0xFFFFFFFF, 0xFFFFFFFF) != 0) {
      printf("Error in xdma agu configuration\n");
  } else {
      // printf("The xdma agu is configured\n");
  }
  task_id = xdma_start();
  xdma_wait(task_id);

 
}

int main() {
  {

    // Create memref objects for data stored in L3
    FourDMemrefI8_t memrefI;
    memrefI.data = &I;
    memrefI.aligned_data = memrefI.data;

    FourDMemrefI8_t memrefW;
    memrefW.data = &W;
    memrefW.aligned_data = memrefW.data;

    TwoDMemrefI8_t memrefB;
    memrefB.data = &B;
    memrefB.aligned_data = memrefB.data;

    FourDMemrefI32_t memrefO;

    // allocate zero row in tcdm
    snrt_l1alloc(256);

    (void)snrt_mcycle();

    _mlir_ciface_toy(&memrefO, &memrefI, &memrefW, &memrefB);

    snrt_cluster_hw_barrier();

    (void)snrt_mcycle();

    // Correctness check -
    // from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0)
      return 0;

    // do not check errors for now, golden model not available

    return 0;
  }
}
