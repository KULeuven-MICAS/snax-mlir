# Courtesy of Federico Ficarelli
FROM rgantonio/snax-chisel-test:latest@sha256:f3d6207ab764a081866886e18f5249891f861061bdf6106d2cf23b6a0342bf96 as deps

RUN git clone https://github.com/kuleuven-micas/snitch_cluster.git /src \
 && cd /src \
 && git reset --hard a276279a6d2d9ca3fa92df050f04bbb21c1b3773 \
 && git submodule update --init

# clang+llvm+lld
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
 # verilator model
RUN  cd /src && make CFG_OVERRIDE=cfg/snax-mac.hjson -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc)
# spike-dasm
RUN cd /src/target/snitch_cluster/work-vlt/riscv-isa-sim \
 && ./configure --prefix=/opt/snitch-spike \
 && make install
 # snitch runtime
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw  -j$(nproc)

FROM deps as snax-gemm
 # verilator model
RUN cd /src && make CFG_OVERRIDE=cfg/snax-gemm.hjson -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc)
 # snitch runtime
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw  -j$(nproc)

FROM ubuntu:18.04 as toolchain
# shared dependencies
COPY --from=deps /opt/snitch-llvm /opt/snitch-llvm
COPY --from=deps /opt/python3.11 /opt/python3.11
COPY --from=deps /src/util/trace /opt/snitch_cluster/util/trace
COPY --from=snax-mac /opt/snitch-spike /opt/snitch-spike

# SNAX-MAC dependencies
COPY --from=snax-mac /src/target/snitch_cluster/bin/snitch_cluster.vlt /opt/snax-mac-rtl/bin/snitch_cluster.vlt
# Compile and link time dependencies
COPY --from=snax-mac /src/sw/snRuntime /opt/snax-mac/sw/snRuntime
COPY --from=snax-mac /src/target/snitch_cluster/sw/runtime/rtl /opt/snax-mac/target/snitch_cluster/sw/runtime/rtl
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
COPY --from=snax-gemm /src/target/snitch_cluster/sw/runtime/rtl /opt/snax-gemm/target/snitch_cluster/sw/runtime/rtl
COPY --from=snax-gemm /src/target/snitch_cluster/sw/runtime/common /opt/snax-gemm/target/snitch_cluster/sw/runtime/common
COPY --from=snax-gemm /src/target/snitch_cluster/sw/snax/ /opt/snax-gemm/target/snitch_cluster/sw/snax
COPY --from=snax-gemm /src/sw/math/ /opt/snax-gemm/sw/math/
# Transitive deps, this stuff must go at some point
COPY --from=snax-gemm /src/sw/deps/riscv-opcodes /opt/snax-gemm/sw/deps/riscv-opcodes
COPY --from=snax-gemm /src/sw/deps/printf /opt/snax-gemm/sw/deps/printf

RUN apt-get -y update \
 && apt-get -y upgrade \
 # python runtime dependencies
 && apt-get -y install \
      wget \
      zlib1g \
      libncurses5 \
      libgdbm5 \
      libnss3 \
      libssl1.1 \
      libreadline7 \
      libffi6 \
      libsqlite3-0 \
      bzip2 \
 # make pip able to install via git
 && apt-get -y install git \
 # mlir
 && apt-get -y install wget lsb-release software-properties-common gnupg \
 && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
 && echo "\
deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic main \n\
deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic main \n\
deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-16 main \n\
deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-16 main\n" >> /etc/apt/sources.list \
 && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
 && apt-get -y update \
 && apt-get -y upgrade \
 && apt-get -y install mlir-16-tools \
 # misc stuff
 && apt-get -y install make \
 #
 && rm -rf /var/lib/apt/lists/* /var/tmp/* /tmp/*

# add python3.11 to path in bashrc
RUN echo "export PATH=/opt/python3.11/bin:$PATH" >> ~/.bashrc

# install python requirements
RUN export PATH=/opt/python3.11/bin:$PATH \
 && git clone https://github.com/kuleuven-micas/snax-mlir.git \
 && cd snax-mlir \
 && pip3 install -r requirements.txt

# install latest xdsl and test dependencies
RUN export PATH=/opt/python3.11/bin:$PATH \
   && pip3 install git+https://github.com/xdslproject/xdsl.git@4c04de0a16df7c3b29f81aa66a7a2603d6b80f1e\
   && pip3 install filecheck \
   && pip3 install lit
