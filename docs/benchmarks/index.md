# Benchmarks

Here you can find the latest results of all relevant Benchmarks

## [Dense Matrix Multiplication](dense_matmul)

Testing peak utilization of the `snax-gemmx` accelerator, for different matrix sizes. Measures max performance without considering control overhead. This test generates a lot of results and plots to get deep insight into the performance of the accelerator combined with the TCDM interconnect and data layout strategies.

The following operations are compared:

- matmul (D = AxB)
- gemm (D = AxB)

The following layout strategies are compared:

- cyclic
- banked

The following sizes are run:

- every combination of (32, 48, 64)
- some common sizes for neural networks
