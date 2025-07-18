- i8 uses assymetric quantization

asssumptions for tosa.rescale:

- double round is always true
- per channel is always false

- verifying the golden model

werkend krijgen op eigen CPU

- lowering through mlir without snax-passes
- mlir-translate and clang
snax-opt rescale_down.mlir -p preprocess --allow-unregistered-dialect
