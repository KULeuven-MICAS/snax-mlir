# Courtesy of Federico Ficarelli

FROM ghcr.io/pulp-platform/snitch_cluster:latest@sha256:9978f9d0e3e6b7ad1b2c1f0eb628a113226e8443bc42ae8b4ddd3f34d9528e96 as builder

RUN git clone --recursive https://github.com/kuleuven-micas/snitch_cluster.git /src \
 # verilator model
 && cd /src/target/snitch_cluster \
 && make bin/snitch_cluster.vlt
 # spike-dasm
RUN cd /src/target/snitch_cluster/work-vlt/riscv-isa-sim \
 && ./configure --prefix=/opt/snitch-spike \
 && make install 
 # snitch runtime
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw 
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
 && mkdir -p /tmp/python-src \
 && wget -qO- https://www.python.org/ftp/python/3.11.5/Python-3.11.5.tgz | \
    tar xz --strip-components=1 -C /tmp/python-src \
 && cd /tmp/python-src \
 && ./configure --prefix=/opt/python3.11 \
 && make install

FROM ubuntu:18.04 as toolchain
COPY --from=builder /src/target/snitch_cluster/bin/snitch_cluster.vlt /opt/snitch-rtl/bin/snitch_cluster.vlt
COPY --from=builder /opt/snitch-spike /opt/snitch-spike
COPY --from=builder /opt/snitch-llvm /opt/snitch-llvm
COPY --from=builder /opt/python3.11 /opt/python3.11
COPY --from=builder /src/util/trace /opt/snitch_cluster/util/trace
# Compile and link time dependencies
COPY --from=builder /src/sw/snRuntime /opt/snitch_cluster/sw/snRuntime
COPY --from=builder /src/target/snitch_cluster/sw/runtime/rtl /opt/snitch_cluster/target/snitch_cluster/sw/runtime/rtl
COPY --from=builder /src/target/snitch_cluster/sw/runtime/common /opt/snitch_cluster/target/snitch_cluster/sw/runtime/common
COPY --from=builder /src/sw/math/ /opt/snitch_cluster/sw/math/
# Transitive deps, this stuff must go at some point
COPY --from=builder /src/sw/deps/riscv-opcodes /opt/snitch_cluster/sw/deps/riscv-opcodes
COPY --from=builder /src/sw/deps/printf /opt/snitch_cluster/sw/deps/printf

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
