# SNAX-MLIR

Repository for running MLIR compiled stuff on SNAX

## Requirements

[stardew](https://github.com/Groverkss/stardew)

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
