rule preprocess_mlir:
    """
    Apply various preprocessing transformations to mlir files with upstream mlir. 
    Options controlled by `mlirpreprocflags` defined in config.
    """
    input:
        "{file}.mlir",
    output:
        temp("{file}.preproc1.mlir"),
        temp("{file}.preproc2.mlir"),
        temp("{file}.preprocfinal.mlir"),
    run:
        shell(
            "{config[mlir-opt]} {config[mlirpreprocflags][0]} -o {wildcards.file}.preproc1.mlir {input}"
        )
        shell(
            "{config[mlir-opt]} {config[mlirpreprocflags][1]} -o {wildcards.file}.preproc2.mlir {wildcards.file}.preproc1.mlir"
        )
        shell(
            "{config[mlir-opt]} {config[mlirpreprocflags][2]} -o {output[2]} {wildcards.file}.preproc2.mlir"
        )


rule snax_opt_mlir:
    """
    Apply various transformations snax-opt on mlir files.
    Options controlled with `snaxoptflags` defined in config.
    """
    input:
        "{file}.preprocfinal.mlir",
    output:
        temp("{file}.snax-opt.mlir"),
    shell:
        "{config[snax-opt]} -p {config[snaxoptflags]} -o {output} {input}"


rule postprocess_mlir:
    """
    Apply various postprocessing transformations to mlir files with upstream mlir.
    Goal is to lower everything to LLVM dialect after this step.
    Options controlled with `mlirpostprocflags` defined in config.
    """
    input:
        "{file}.snax-opt.mlir",
    output:
        temp("{file}.ll.mlir"),
    shell:
        "{config[mlir-opt]} {config[mlirpostprocflags]} -o {output} {input}"


rule postprocess_no_snax_mlir:
    """
    Bypass snax-opt, and apply various postprocessing transformations to mlir files with upstream mlir.
    Goal is to lower everything to LLVM dialect after this step.
    Options controlled with `mlirpostprocflags` defined in config.
    """
    input:
        "{file}.preprocfinal.mlir",
    output:
        temp("{file}.no-snax-opt.ll.mlir"),
    shell:
        "{config[mlir-opt]} {config[mlirpostprocflags]} -o {output} {input}"


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


rule postprocess_llvm_module:
    """
    Add extra metadata to LLVM module required for snitch-based systems
    """
    input:
        "{file}.ll",
    output:
        temp("{file}.ll12"),
    shell:
        "../../runtime/tollvm12.py < {input} > {output} "


rule compile_llvm_module:
    """
    Use clang to compile LLVM module to object file.
    Uses target-specific options, but not C-specific options.
    """
    input:
        "{file}.ll12",
    output:
        temp("{file}.o"),
    shell:
        "{config[cc]} {config[clangflags]} -x ir -c {input} -o {output}"


rule clean:
    """
    Remove generated files.
    """
    shell:
        "rm -rf *.ll12 *.x *.o *.logs/ logs/ data* *.dasm"
