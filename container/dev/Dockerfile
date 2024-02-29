FROM ghcr.io/kuleuven-micas/snax-mlir:main as deps

FROM ubuntu:20.04 as devcontainer

# shared dependencies
COPY --from=deps /opt /opt

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
    && apt-get install -y python3.11 python3.11-distutils \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10 \
    && apt-get install curl git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN git clone https://github.com/kuleuven-micas/snax-mlir.git \
 && cd snax-mlir \
 && python3 -m pip install -r requirements.txt
