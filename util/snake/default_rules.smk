rule convert_tflite_to_tosa:
    input:
        "{file}.tflite",
    output:
        temp("{file}.mlir.bc"),
    shell:
        "../../runtime/tflite_to_tosa.py -c {input} -o {output} "


rule convert_mlir_bytecode_to_text:
    input:
        "{file}.mlir.bc",
    output:
        temp("{file}.mlir"),
    shell:
        "{config[mlir-opt]} --mlir-print-op-generic --mlir-print-local-scope -o {output} {input}"


rule preprocess_mlir:
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
    input:
        "{file}.preprocfinal.mlir",
    output:
        temp("{file}.snax-opt.mlir"),
    shell:
        "{config[snax-opt]} -p {config[snaxoptflags]} -o {output} {input}"


rule postprocess_mlir:
    input:
        "{file}.snax-opt.mlir",
    output:
        temp("{file}.ll.mlir"),
    shell:
        "{config[mlir-opt]} {config[mlirpostprocflags]} -o {output} {input}"


rule postprocess_no_snax_mlir:
    input:
        "{file}.preprocfinal.mlir",
    output:
        temp("{file}.no-snax-opt.ll.mlir"),
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
        "{config[cc]} {config[clangflags]} -x ir -c {input} -o {output}"


rule postprocess_llvm_module:
    input:
        "{file}.ll",
    output:
        temp("{file}.ll12"),
    shell:
        "../../runtime/tollvm12.py < {input} > {output} "


rule generate_simple_data:
    output:
        "data.c",
        "data.h",
    script:
        "gendata.py"


rule compile_simple_main:
    input:
        "main.c",
        "data.o",
    output:
        temp("main.o"),
    shell:
        "{config[cc]} {config[cflags]} -c {input} -o {output}"


rule rtl_simulation:
    input:
        "{file}.x",
    output:
        "logs/trace_chip_{numchips}_hart_{hartid}.dasm",
    shell:
        "{config[vltsim]} {input} --trace-prefix {file}"


rule clean:
    shell:
        "rm -rf *.ll12 *.x *.o *.logs/ logs/ data* *.dasm"
