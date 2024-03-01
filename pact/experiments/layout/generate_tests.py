import os
import numpy as np

directory = os.path.dirname(__file__)

test_cases = [
    [16, 16, 16], # ops = 16*16*16 = 4096
    [16, 16, 32], # ops = 16*16*32 = 8192
    [16, 32, 32], # ops = 16*32*32 = 16384
    [32, 32, 32], # ops = 32*32*32 = 32768
    [32, 32, 64], # ops = 32*32*64 = 65536
    [32, 64, 64], # ops = 32*64*64 = 131072
    [64, 64, 64], # ops = 64*64*64 = 262144
    [64, 64, 128], # ops = 64*64*128 = 524288
    [64, 128, 128], # ops = 64*128*128 = 1048576
    [128, 128, 128], # ops = 128*128*128 = 2097152
]

def generate_mlir(testcase):
    memref_a = f'memref<{testcase[0]}x{testcase[1]}xi32>'
    memref_b = f'memref<{testcase[1]}x{testcase[2]}xi32, strided<[1, {testcase[1]}], offset:0>>'
    memref_c = f'memref<{testcase[0]}x{testcase[2]}xi32>'

    template_path = os.path.join(directory, 'mlir_template.mlir')
    template = open(template_path).read()
    return template.format(memref_a=memref_a, memref_b=memref_b, memref_c=memref_c)

def generate_main(testcase, layout):

    if layout == 'default':
        strideInnermostA = 1,
        strideInnermostB = 1,
        strideInnermostC = 1,
        ldA = 1,
        ldB = 1,
        ldC = 1,
    elif layout == 'tiled':
        strideInnermostA = 1,
        strideInnermostB = 1,
        strideInnermostC = 1,
        ldA = 1,
        ldB = 1,
        ldC = 1,
    else:
        raise ValueError(f'Unknown layout: {layout}')


    template_path = os.path.join(directory, 'main_template.cpp')
    template = open(template_path).read()
    return template.format(
        strideInnermostA = strideInnermostA,
        strideInnermostB = strideInnermostB,
        strideInnermostC = strideInnermostC,
        ldA = ldA,
        ldB = ldB,
        ldC = ldC,
    )

def main():

    for i, testcase in enumerate(test_cases):

        mlir = generate_mlir(testcase)
        main = generate_main(testcase, 'default')
    
    generate_mlir()

if __name__ == "__main__":
    main()

