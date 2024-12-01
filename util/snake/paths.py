def get_default_paths(snitch_llvm_path="/usr/bin"):
    cc_path = f"{snitch_llvm_path}/clang"
    return {
        "cc": cc_path,
        "ld": cc_path,  # LD is also CC
        "mlir-opt": "mlir-opt",
        "mlir-translate": "mlir-translate",
        "snax-opt": "snax-opt",
    }
