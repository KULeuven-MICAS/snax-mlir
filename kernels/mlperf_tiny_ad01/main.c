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


void _mlir_ciface_run_network(TwoDMemrefI8_t *output, TwoDMemrefI8_t *input);

void _mlir_ciface_snax_debug_gemm(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c, int32_t when) {
  int8_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int8_t*) _ptr_a;
  ptr_b = (int8_t*) _ptr_b;
  ptr_c = (int8_t*) _ptr_c;

  printf("Debugging GeMM at t = %d with A at %p, B at %p, C at %p\n", when, ptr_a, ptr_b, ptr_c);

}

void _mlir_ciface_snax_debug_bias(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c, int32_t when) {
  int8_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int8_t*) _ptr_a;
  ptr_b = (int8_t*) _ptr_b;
  ptr_c = (int8_t*) _ptr_c;

  printf("Debugging bias at t = %d with A at %p, B at %p, C at %p\n", when, ptr_a, ptr_b, ptr_c);

}

void _mlir_ciface_snax_debug_simd(int32_t _ptr_a, int32_t _ptr_b, int32_t _ptr_c, int32_t when) {
  int8_t *ptr_a, *ptr_b, *ptr_c;
  ptr_a = (int8_t*) _ptr_a;
  ptr_b = (int8_t*) _ptr_b;
  ptr_c = (int8_t*) _ptr_c;

  printf("Debugging SIMD at t = %d with A at %p, B at %p, C at %p\n", when, ptr_a, ptr_b, ptr_c);

}

int main() {
  {

    // Create memref objects for data stored in L3
    TwoDMemrefI8_t memrefA;
    memrefA.data = &A;
    memrefA.aligned_data = memrefA.data;
    // Shape and Stride need to be defined for dynamic case
    memrefA.shape[0] = 8;
    memrefA.shape[1] = 640;
    memrefA.stride[0] = 640;
    memrefA.stride[1] = 1;
    memrefA.offset = 0;

    TwoDMemrefI8_t memrefB;

    (void)snrt_mcycle();

    _mlir_ciface_run_network(&memrefB, &memrefA);

    snrt_cluster_hw_barrier();

    (void)snrt_mcycle();

    return 0;
  }
}
