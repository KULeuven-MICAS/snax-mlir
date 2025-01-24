rule snax_opt_mlir:
    """
    Apply various transformations snax-opt on mlir files.
    Options controlled with `snaxoptflags` defined in config.
    """
    input:
        "{file}.mlir",
    output:
        temp("{file}.ll.mlir"),
    shell:
        "{config[snax-opt]} -p {config[snaxoptflags]} -o {output} {input}"


rule translate_mlir:
    """
    Translate MLIR LLVM dialect to actual LLVM.
    """
    input:
        "{file}.ll.mlir",
    output:
        temp("{file}.ll"),
    shell:
        "{config[mlir-translate]} --mlir-to-llvmir -o {output} {input}"


rule compile_c:
    """
    Generic rule to compile c files with default compilation options.
    """
    input:
        "{file}.c",
    output:
        temp("{file}.o"),
    shell:
        "{config[cc]} {config[cflags]} -c {input} -o {output}"


rule compile_llvm_module:
    """
    Use clang to compile LLVM module to object file.
    Uses target-specific options, but not C-specific options.
    """
    input:
        "{file}.ll",
    output:
        temp("{file}.o"),
    shell:
        "{config[cc]} {config[clangflags]} -x ir -c {input} -o {output}"


rule trace_dasm:
    """
    Use spike-dasm and gen_trace.py to make simulation traces human-readable
    and aggregate stats for a specific hart's trace.
    """
    input:
        "{file}.dasm",
    output:
        temp("{file}.json"),
        temp("{file}.txt"),
    shell:
        "{config[cc]} {config[clangflags]} -x ir -c {input} -o {output}"


rule clean:
    """
    Remove generated files.
    """
    shell:
        "rm -rf *.ll12 *.x *.o *.logs/ logs/ data* *.dasm"
