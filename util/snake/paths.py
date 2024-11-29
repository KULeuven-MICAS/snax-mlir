def get_default_paths(snitch_llvm_path):
    return [
        CC := f"{snitch_llvm_path}/clang",
        CC,  # LD is also CC
        "mlir-opt",
        "mlir-translate",
        "snax-opt",
    ]
