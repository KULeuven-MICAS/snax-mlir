# Courtesy of Federico Ficarelli
FROM ghcr.io/kuleuven-micas/snax:latest as deps

ARG SNITCH_BRANCH=fifo-2

RUN apt-get update && apt-get install -y curl

RUN sbt package

RUN git clone https://github.com/jorendumoulin/pact24_snitch_cluster.git /src \
 && cd /src \
 && git checkout ${SNITCH_BRANCH} \
 && git submodule update --init

FROM deps as snax-streamer-gemm

# verilator model
RUN  cd /src && \ 
     make  -C target/snitch_cluster bin/snitch_cluster.vlt -j$(nproc)  CFG_OVERRIDE=cfg/snax-streamer-gemm.hjson

# runtime and apps
RUN cd /src/target/snitch_cluster \
 && make DEBUG=ON sw  -j$(nproc) \
 SELECT_TOOLCHAIN=llvm-generic \
 SELECT_RUNTIME=rtl-generic \
 CFG_OVERRIDE=cfg/snax-streamer-gemm.hjson

FROM ubuntu:18.04 as toolchain

COPY --from=snax-streamer-gemm /src/target/snitch_cluster/bin/snitch_cluster.vlt /opt/snax-streamer-gemm-rtl/bin/snitch_cluster.vlt
COPY --from=snax-streamer-gemm /src/sw/snRuntime /opt/snax-streamer-gemm/sw/snRuntime
COPY --from=snax-streamer-gemm /src/target/snitch_cluster/sw/runtime/rtl /opt/snax-streamer-gemm/target/snitch_cluster/sw/runtime/rtl
COPY --from=snax-streamer-gemm /src/target/snitch_cluster/sw/runtime/rtl-generic /opt/snax-streamer-gemm/target/snitch_cluster/sw/runtime/rtl-generic
COPY --from=snax-streamer-gemm /src/target/snitch_cluster/sw/runtime/common /opt/snax-streamer-gemm/target/snitch_cluster/sw/runtime/common
COPY --from=snax-streamer-gemm /src/target/snitch_cluster/sw/snax/ /opt/snax-streamer-gemm/target/snitch_cluster/sw/snax
COPY --from=snax-streamer-gemm /src/target/snitch_cluster/sw/apps/ /opt/snax-streamer-gemm/target/snitch_cluster/sw/apps

COPY --from=snax-streamer-gemm /src/sw/math/ /opt/snax-streamer-gemm/sw/math/
