import os
from collections.abc import Sequence


def get_target_flags() -> Sequence[str]:
    """
    Function that returns llvm target flags, related to RISC-V backend settings
    """
    return [
        "--target=riscv64-unknown-elf",
        "-mcpu=generic-rv64",
        "-march=rv64imafdc",
        "-mabi=lp64d",
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


def get_cc_flags(cva6_sw_path: str) -> Sequence[str]:
    """
    Function that returns default c-compiler flags
    """
    return [
        f"-I{cva6_sw_path}/target/sw/host/runtime",
        f"-I{cva6_sw_path}/target/sw/shared/runtime",
        f"-I{cva6_sw_path}/target/sw/shared/platform/generated",
        *get_clang_flags(),
    ]


def get_ld_flags(cva6_sw_path: str) -> Sequence[str]:
    """
    Function that returns default linker flags
    """
    return [
        *get_target_flags(),
        f"-T{cva6_sw_path}/target/sw/host/runtime/host.ld",
        "-nostdlib",
    ]


def get_cva6_flags(cva6_sw_path: str):
    return {
        "cflags": get_cc_flags(cva6_sw_path),
        "clangflags": get_clang_flags(),
        "ldflags": get_ld_flags(cva6_sw_path),
    }
