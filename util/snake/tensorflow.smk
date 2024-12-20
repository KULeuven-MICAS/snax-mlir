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
