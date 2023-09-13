#include "data.h"
#include "stdint.h"

#include <snrt.h>

// Kernel provided via external definition
void simple_mult(int32_t *a, int32_t *b, int32_t *d);

int main() {
    // Allocate shared local memory
    // By avoiding allocators and bumping by a known offset a base pointer
    // (snrt_l1_next()) that is the same for all the cores in the cluster, we are
    // essentially providing the same memory regions to all the cores in this cluster.
    int32_t *local_A = (int32_t*)snrt_l1_next();
    int32_t *local_B = local_A + N;
    int32_t *local_D = local_B + N;

    // Copy data in shared local memory
    if (snrt_is_dm_core()) {
        snrt_dma_start_1d(local_A, A, N * sizeof(float));
        snrt_dma_start_1d(local_B, B, N * sizeof(float));
    }

    snrt_cluster_hw_barrier();

    // Launch kernel: from this point on only core 0 is required to be alive.
    int thiscore = snrt_cluster_core_idx();
    if (thiscore != 0) return 0;

    (void)snrt_mcycle();
    simple_mult(local_A, local_B, local_D);
    (void)snrt_mcycle();

    // Correctness check
    int nerr = 0;
    for (int i = 0; i < N; i++) {
        int32_t error = local_D[i] - G[i];
        if (error != 0) nerr+=1;   
    }
    return nerr;
}
