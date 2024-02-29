# Courtesy of Federico Ficarelli
FROM ghcr.io/kuleuven-micas/snax:latest as deps

RUN git clone https://github.com/kuleuven-micas/snitch_cluster.git /src \
 && cd /src \
 && git reset --hard d87720fdf66b15325b37391494b990ac606503ab \
 && git submodule update --init

# Snitch-specific version for libc and clang runtime libraries
# LLVM 17 is already installed in base image
RUN mkdir -p /opt/snitch-llvm \
 && wget -qO- https://github.com/pulp-platform/llvm-project/releases/download/0.12.0/riscv32-pulp-llvm-ubuntu1804-0.12.0.tar.gz | \
    tar xz --strip-components=1 -C /opt/snitch-llvm
# python 3.11
RUN apt update -y \
 && apt install -y \
      build-essential \
      zlib1g-dev \
      libncurses5-dev \
      libgdbm-dev \
      libnss3-dev \
      libssl-dev \
      libreadline-dev \
      libffi-dev \
      libsqlite3-dev \
      wget \
      libbz2-dev \
      lzma \
      liblzma-dev \
      libbz2-dev \
 && mkdir -p /tmp/python-src \
 && wget -qO- https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz | \
    tar xz --strip-components=1 -C /tmp/python-src \
 && cd /tmp/python-src \
 && ./configure --prefix=/opt/python3.11 \
 && make install

FROM deps as snax-mac
# snitch runtime
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw  -j$(nproc) \
 CFG_OVERRIDE=cfg/snax-mac.hjson \
 SELECT_TOOLCHAIN=llvm-generic \
 SELECT_RUNTIME=rtl-generic
# verilator model
RUN  cd /src && \
make  -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) \
 CFG_OVERRIDE=cfg/snax-mac.hjson
# spike-dasm
RUN cd /src/target/snitch_cluster/work-vlt/riscv-isa-sim \
 && ./configure --prefix=/opt/snitch-spike \
 && make install

FROM deps as snax-gemm
# snitch runtime
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw  -j$(nproc) \
 SELECT_TOOLCHAIN=llvm-generic \
 SELECT_RUNTIME=rtl-generic \
 CFG_OVERRIDE=cfg/snax-gemm.hjson
# verilator model
RUN  cd /src && \
 make  -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc) \
 CFG_OVERRIDE=cfg/snax-gemm.hjson

FROM ubuntu:20.04 as toolchain
# shared dependencies
# libc for memset from old toolchain
COPY --from=deps /opt/snitch-llvm/riscv32-unknown-elf/lib /opt/snitch-llvm/riscv32-unknown-elf/lib
# libclang_rt.builtins_riscv32 from old toolchain
COPY --from=deps /opt/snitch-llvm/lib/clang/12.0.1/lib /opt/snitch-llvm/lib/clang/12.0.1/lib
COPY --from=deps /opt/python3.11 /opt/python3.11
COPY --from=deps /src/util/trace /opt/snitch_cluster/util/trace
COPY --from=snax-mac /opt/snitch-spike /opt/snitch-spike

# SNAX-MAC dependencies
COPY --from=snax-mac /src/target/snitch_cluster/bin/snitch_cluster.vlt /opt/snax-mac-rtl/bin/snitch_cluster.vlt
# Compile and link time dependencies
COPY --from=snax-mac /src/sw/snRuntime /opt/snax-mac/sw/snRuntime
# rtl-generic includes from rtl, so we also need rtl, even if we don't build it
COPY --from=snax-mac /src/target/snitch_cluster/sw/runtime/rtl /opt/snax-mac/target/snitch_cluster/sw/runtime/rtl
COPY --from=snax-mac /src/target/snitch_cluster/sw/runtime/rtl-generic /opt/snax-mac/target/snitch_cluster/sw/runtime/rtl-generic
COPY --from=snax-mac /src/target/snitch_cluster/sw/runtime/common /opt/snax-mac/target/snitch_cluster/sw/runtime/common
COPY --from=snax-mac /src/target/snitch_cluster/sw/snax/ /opt/snax-mac/target/snitch_cluster/sw/snax
COPY --from=snax-mac /src/sw/math/ /opt/snax-mac/sw/math/
# Transitive deps, this stuff must go at some point
COPY --from=snax-mac /src/sw/deps/riscv-opcodes /opt/snax-mac/sw/deps/riscv-opcodes
COPY --from=snax-mac /src/sw/deps/printf /opt/snax-mac/sw/deps/printf

# SNAX-GEMM dependencies
COPY --from=snax-gemm /src/target/snitch_cluster/bin/snitch_cluster.vlt /opt/snax-gemm-rtl/bin/snitch_cluster.vlt
# Compile and link time dependencies
COPY --from=snax-gemm /src/sw/snRuntime /opt/snax-gemm/sw/snRuntime
# rtl-generic includes from rtl, so we also need rtl, even if we don't build it
COPY --from=snax-gemm /src/target/snitch_cluster/sw/runtime/rtl /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl
COPY --from=snax-gemm /src/target/snitch_cluster/sw/runtime/rtl-generic /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl-generic
COPY --from=snax-gemm /src/target/snitch_cluster/sw/runtime/common /opt/snax-gemm/target/snitch_cluster/sw/runtime/common
COPY --from=snax-gemm /src/target/snitch_cluster/sw/snax/ /opt/snax-gemm/target/snitch_cluster/sw/snax
COPY --from=snax-gemm /src/sw/math/ /opt/snax-gemm/sw/math/
# Transitive deps, this stuff must go at some point
COPY --from=snax-gemm /src/sw/deps/riscv-opcodes /opt/snax-gemm/sw/deps/riscv-opcodes
COPY --from=snax-gemm /src/sw/deps/printf /opt/snax-gemm/sw/deps/printf

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y software-properties-common \
    && apt-get -y install wget lsb-release gnupg \
    ## llvm & mlir
    && wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 17 && rm llvm.sh \
    && apt-get -y install mlir-17-tools \
    && apt-get -y install make \
    ## python3
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get install -y python3.11-full \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10 \
    && apt-get install -y curl git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN git clone https://github.com/kuleuven-micas/snax-mlir.git \
 && cd snax-mlir \
 && python3 -m pip install -r requirements.txt
