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
