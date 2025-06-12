import os
from collections.abc import Sequence


def get_target_flags() -> Sequence[str]:
    """
    Function that returns llvm target flags, related to RISC-V backend settings
    """
    return [
        "--target=riscv32-unknown-elf",
        "-mcpu=generic-rv32",
        "-march=rv32imafdzfh",
        "-mabi=ilp32d",
        "-mcmodel=medany",
    ]


def get_clang_flags() -> Sequence[str]:
    """
    Function that returns clang-specific flags, related to RISC-V backend settings
    """
    return [
        "-Wno-unused-command-line-argument",
        *get_target_flags(),
        "-ftls-model=local-exec",
        "-ffast-math",
        "-fno-builtin-printf",
        "-fno-common",
        "-O3",
        "-std=gnu11",
        "-Wall",
        "-Wextra",
    ]


def get_cc_flags(snitch_sw_path: str, snax_mlir_path: str) -> Sequence[str]:
    """
    Function that returns default c-compiler flags
    """
    return [
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
        f"-I{snax_mlir_path}/runtime/include",
        "-D__DEFINED_uint64_t",
        *get_clang_flags(),
    ]


def get_ld_flags(snitch_sw_path: str, snitch_llvm_path: str | None = None) -> Sequence[str]:
    """
    Function that returns default linker flags
    """
    # Default path points to conda/pixi environment
    if snitch_llvm_path is None:
        snitch_llvm_path = os.environ["CONDA_PREFIX"] + "/bin"
    return [
        f"-fuse-ld={snitch_llvm_path}/ld.lld",
        *get_target_flags(),
        f"-T{snitch_sw_path}/sw/snRuntime/base.ld",
        f"-L{snitch_sw_path}/target/snitch_cluster/sw/runtime/rtl-generic",
        f"-L{snitch_sw_path}/target/snitch_cluster/sw/runtime/rtl-generic/build",
        "-nostdlib",
        "-lsnRuntime",
    ]


def get_default_flags(
    snitch_sw_path: str,
    snitch_llvm_path: str | None = None,
    snax_mlir_path: str | None = None,
):
    if snitch_llvm_path is None:
        snitch_llvm_path = os.environ["CONDA_PREFIX"] + "/bin"
    if snax_mlir_path is None:
        snax_mlir_path = "../.."
    return {
        "cflags": get_cc_flags(snitch_sw_path, snax_mlir_path),
        "clangflags": get_clang_flags(),
        "ldflags": get_ld_flags(snitch_sw_path, snitch_llvm_path),
    }
