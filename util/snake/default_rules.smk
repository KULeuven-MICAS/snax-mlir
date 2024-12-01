rule postprocess_mlir:
    input:
        "{file}.snax-opt.mlir",
    output:
        temp("{file}.ll.mlir"),
    shell:
        "{config[mlir-opt]} {config[mlirpostprocflags]} -o {output} {input}"


rule translate_mlir:
    input:
        "{file}.ll.mlir",
    output:
        temp("{file}.ll"),
    shell:
        "{config[mlir-translate]} --mlir-to-llvmir -o {output} {input}"


rule compile_c:
    input:
        "{file}.c",
    output:
        temp("{file}.o"),
    shell:
        "{config[cc]} {config[cflags]} -c {input} -o {output}"


rule compile_llvm_module:
    input:
        "{file}.ll12",
    output:
        temp("{file}.o"),
    shell:
        "{config[cc]} {config[cflags]} -x ir -c {input} -o {output}"


rule postprocess_llvm_module:
    input:
        "{file}.ll",
    output:
        temp("{file}.ll12"),
    shell:
        "../../runtime/tollvm12.py < {input} > {output} "


rule clean:
    shell:
        "rm -rf *.ll12 *.x *.o *.logs/ logs/ data*"
