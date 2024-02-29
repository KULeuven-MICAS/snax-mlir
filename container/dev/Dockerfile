FROM ghcr.io/kuleuven-micas/snax-mlir:main as deps

FROM ubuntu:22.04 as devcontainer

# shared dependencies
COPY --from=deps /opt /opt
