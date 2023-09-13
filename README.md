# SNAX-MLIR

Repository for running MLIR compiled stuff on SNAX

## Requirements

* [stardew](https://github.com/Groverkss/stardew)
* numpy

## Docker Container

For compiling the low-level kernels you need the snitch compilation toolchain, 
which is easiest to get through a docker container.
A Dockerfile for such container is provided here (note that leaving out the `config` `--build-arg` will use the default `snitch_cluster` setup:
```sh
cd container
docker build . -t snax-toolchain --build-arg config=path_to_your_hjson_file.hjson
cd ..
```
Then you can run your experiments in this repository with:
```sh
docker run -itv `pwd`:/repo:z snax-toolchain
```
The repository will be available under `/repo`

## Run Kernels

Inside the docker container:
```sh
cd /kernels/simple_mult
make allrun
```
This will compile `main.c` two different `kernel`s:

1. `baseline.c`: A C implementation of the kernel
2. `linalg.mlir`: An MLIR Linalg implementation of the kernel

Note that for both kernels, a different lowering path is employed.
All C code is lowered with the same flow (1):

1. `c code` -> `clang-12` (snitch-specific) -> RISC-V binary -> `ld.lld-12` -> RISC-V executable
2. `linalg code` -> `mlir-opt-16` (16.0.6) -> `llvm` dialect -> `mlir-translate-16` (16.0.6) -> `llvm` bytecode -> `tollvm12.py` -> `llvm-12` bytecode -> `clang-12` (snitch-specific) -> RISC-V binary -> `ld.lld-12` -> RISC-V executable
