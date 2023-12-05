#include <stdint.h>
#pragma once

#define N_size 16
#define K_size 16
#define M_size 16
extern const int8_t A[N_size * K_size];
extern const int8_t B[K_size * M_size];
extern const int32_t G[N_size * M_size];
