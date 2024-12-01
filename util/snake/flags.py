def get_mlir_preproc_flags():
    return [
        [
            "--pass-pipeline='builtin.module(func.func("
            + ", ".join(
                [
                    "tosa-to-linalg-named",
                    "tosa-to-tensor",
                    "tosa-to-scf",
                    "tosa-to-linalg",
                ]
            )
            + "))'",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
        ],
        ["--tosa-to-arith='include-apply-rescale'", "--empty-tensor-to-alloc-tensor"],
        [
            "--test-linalg-transform-patterns='test-generalize-pad-tensor'",
            "--linalg-generalize-named-ops",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize='"
            + " ".join(
                [
                    "bufferize-function-boundaries",
                    "allow-return-allocs-from-loops",
                    "function-boundary-type-conversion=identity-layout-map",
                ]
            )
            + "'",
            "--mlir-print-op-generic",
            "--mlir-print-local-scope",
        ],
    ]


def get_mlir_postproc_flags(index_bitwidth=32):
    return [
        "--convert-linalg-to-loops",
        "--convert-scf-to-cf",
        "--lower-affine",
        "--canonicalize",
        "--cse",
        "--convert-math-to-llvm",
        "--llvm-request-c-wrappers",
        "--expand-strided-metadata",
        "--lower-affine",
        f"--convert-index-to-llvm=index-bitwidth={index_bitwidth}",
        f"--convert-cf-to-llvm=index-bitwidth={index_bitwidth}",
        f"--convert-arith-to-llvm=index-bitwidth={index_bitwidth}",
        f"--convert-func-to-llvm='index-bitwidth={index_bitwidth}'",
        f"--finalize-memref-to-llvm='use-generic-functions index-bitwidth={index_bitwidth}'",
        "--canonicalize",
        "--reconcile-unrealized-casts",
    ]


def get_cc_flags(snitch_sw_path):
    return [
        "-Wno-unused-command-line-argument",
        f"-I{snitch_sw_path}/target/snitch_cluster/sw/runtime/rtl-generic/src",
        f"-I{snitch_sw_path}/target/snitch_cluster/sw/runtime/common",
        f"-I{snitch_sw_path}/sw/snRuntime/api",
        f"-I{snitch_sw_path}/sw/snRuntime/src",
        f"-I{snitch_sw_path}/sw/snRuntime/src/omp/",
        f"-I{snitch_sw_path}/sw/snRuntime/api/omp/",
        f"-I{snitch_sw_path}/sw/math/arch/riscv64/bits/",
        f"-I{snitch_sw_path}/sw/math/arch/generic",
        f"-I{snitch_sw_path}/sw/math/src/include",
        f"-I{snitch_sw_path}/sw/math/src/internal",
        f"-I{snitch_sw_path}/sw/math/include/bits",
        f"-I{snitch_sw_path}/sw/math/include",
        "-I../../runtime/include",
        "-D__DEFINED_uint64_t",
        "--target=riscv32-unknown-elf",
        "-mcpu=generic-rv32",
        "-march=rv32imafdzfh",
        "-mabi=ilp32d",
        "-mcmodel=medany",
        "-ftls-model=local-exec",
        "-ffast-math",
        "-fno-builtin-printf",
        "-fno-common",
        "-O3",
        "-std=gnu11",
        "-Wall",
        "-Wextra",
    ]


def get_ld_flags(snitch_sw_path, snitch_llvm_path="/usr/bin"):
    return [
        f"-fuse-ld={snitch_llvm_path}/ld.lld",
        "--target=riscv32-unknown-elf",
        "-mcpu=generic-rv32",
        "-march=rv32imafdzfh",
        "-mabi=ilp32d",
        "-mcmodel=medany",
        f"-T{snitch_sw_path}/sw/snRuntime/base.ld",
        f"-L{snitch_sw_path}/target/snitch_cluster/sw/runtime/rtl-generic",
        f"-L{snitch_sw_path}/target/snitch_cluster/sw/runtime/rtl-generic/build",
        "-nostdlib",
        "-lsnRuntime",
    ]


def get_default_flags(snitch_sw_path, snitch_llvm_path="/usr/bin", index_bitwidth=32):
    return {
        "cflags": get_cc_flags(snitch_sw_path),
        "ldflags": get_ld_flags(snitch_sw_path, snitch_llvm_path),
        "mlirpreprocflags": get_mlir_preproc_flags(),
        "mlirpostprocflags": get_mlir_postproc_flags(index_bitwidth),
    }
