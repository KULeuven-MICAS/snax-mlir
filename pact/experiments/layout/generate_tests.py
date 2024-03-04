import os
import numpy as np
import numpy.typing as npt
import itertools

directory = os.path.dirname(__file__)

# size is defined as [M, N, K]
# A is MxN, B is NxK, C is MxK
sizes = [
    [16, 16, 16], # ops = 16*16*16 = 4096
    # [16, 16, 32], # ops = 16*16*32 = 8192
    # [16, 32, 32], # ops = 16*32*32 = 16384
    # [32, 32, 32], # ops = 32*32*32 = 32768
    # [32, 32, 64], # ops = 32*32*64 = 65536
    # [32, 64, 64], # ops = 32*64*64 = 131072
    # [64, 64, 64], # ops = 64*64*64 = 262144
    # [64, 64, 128], # ops = 64*64*128 = 524288
    # [64, 128, 128], # ops = 64*128*128 = 1048576
    # [128, 128, 128], # ops = 128*128*128 = 2097152
]

layouts = [
    # 'default',
    'tiled'
]

backends = [
    'cpu',      # cpu golden model
    'base',     # base system (no streamers)
    'fifo-2',   # streamer with a fifo depth of 2
]

def generate_mlir(size):
    memref_a = f'memref<{size[0]}x{size[1]}xi8>'
    memref_b = f'memref<{size[1]}x{size[2]}xi8, strided<[1, {size[1]}], offset:0>>'
    memref_c = f'memref<{size[0]}x{size[2]}xi32>'

    template_path = os.path.join(directory, 'mlir_template.mlir')
    template = open(template_path).read()
    return template.format(memref_a=memref_a, memref_b=memref_b, memref_c=memref_c)

def generate_main(size, layout, backend):

    if layout == 'default':
        raise ValueError('Not yet implemented')
        strideInnermostA = 1,
        strideInnermostB = 1,
        strideInnermostC = 1,
        ldA = 1,
        ldB = 1,
        ldC = 1,
    elif layout == 'tiled':
        strideInnermostA = 256
        strideInnermostB = 256
        strideInnermostC = 256
        ldA = round(256 * size[1] // 8)
        ldB = round(256 * size[1] // 8)
        ldC = round(256 * size[2] // 8)
    else:
        raise ValueError(f'Unknown layout: {layout}')

    if backend == 'cpu':
        template_path = os.path.join(directory, 'main_template_cpu.c')
    elif backend == 'base':
        template_path = os.path.join(directory, 'main_template_base.c')
    elif backend == 'fifo-2':
        template_path = os.path.join(directory, 'main_template_fifo.c')
    else:
        raise ValueError(f'Unsupported backend: {backend}')
    
    template = open(template_path).read()
    return template.format(
        strideInnermostA = strideInnermostA,
        strideInnermostB = strideInnermostB,
        strideInnermostC = strideInnermostC,
        ldA = ldA,
        ldB = ldB,
        ldC = ldC,
    )
    
    
def generate_makefile(backend):

    if backend in ['cpu', 'base']:
        runtime_backend = 'snax-gemm'
    elif backend == 'fifo-2':
        runtime_backend = 'snax-streamer-gemm-fifo-2'
    else:
        raise ValueError(f'Unsupported backend: {backend}')

    template_path = os.path.join(directory, 'makefile_template')
    template = open(template_path).read()
    return template.format(
        backend = runtime_backend,
    )

def create_header(sizes: dict[str, int], variables: dict[str, npt.NDArray]) -> None:
    includes = ["#include <stdint.h>", "#pragma once", ""]
    includes = "\n".join(includes)
    variables_string = [""]
    for i, j in sizes.items():
        variables_string.append(f"#define {i} {j}")
    variables_string.append("")
    for i, j in variables.items():
        variables_string.append(f"extern const {j.dtype}_t {i}[{j.size}];")
    variables_string = "\n".join(variables_string)
    return "\n".join([includes, variables_string])


def create_data(variables: dict[str, npt.NDArray]):
    includes = ['#include "data.h"', "", ""]
    result = "\n".join(includes)
    variables = {i: np.reshape(j, j.size) for i, j in variables.items()}

    for variable_name, variable_value in variables.items():
        result += f"const {variable_value.dtype}_t {variable_name}" + f"[{variable_value.size}] = " + "{\n"
        variable_str = ["\t" + str(i) for i in variable_value]
        result += ",\n".join(variable_str)
        result += "\n};\n\n"

    return result
    
def generate_data(size):

    low_bound = -128
    high_bound = 127
    A_size = [size[0], size[1]]
    B_size = [size[1], size[2]]
    np.random.seed(0)

    # C = A.B
    A = np.random.randint(low_bound, high_bound, size=A_size, dtype=np.dtype("int8"))
    B = np.random.randint(low_bound, high_bound, size=B_size, dtype=np.dtype("int8"))
    # Make sure the product is possible!
    assert A.shape[1] == B.shape[0]
    C_golden = np.matmul(A.astype(np.dtype("int32")), B.astype(np.dtype("int32")))
    C = np.zeros(C_golden.shape, np.dtype("int32"))

    sizes = {"N_size": A.shape[0], "K_size": A.shape[1], "M_size": B.shape[1]}

    # Perform layout transformations before writing to memory

    # only thing necessary: transform B from row-major to column-major
    B_new_layout = np.transpose(B)

    # C are just all zeros, so layout not important
    variables = {
        "A": A,
        "B": B_new_layout,
        "C_golden": C_golden,
        "C": C,
    }

    header = create_header(sizes, variables)
    data = create_data(variables)

    return header, data


def main():

    test_cases = list(itertools.product(sizes, layouts, backends))

    for i, testcase in enumerate(test_cases):

        size, layout, backend = testcase

        mlir = generate_mlir(size)
        main = generate_main(size, layout, backend)
        makefile = generate_makefile(backend)
        header, data = generate_data(size)

        test_dir = os.path.join(directory, f'test_{i}_{layout}_{backend}_{"_".join(map(str, size))}')
        os.makedirs(test_dir, exist_ok=True)

        # write all the files
        with open(os.path.join(test_dir, 'matmul.mlir'), 'w') as f:
            f.write(mlir)
        with open(os.path.join(test_dir, 'main.c'), 'w') as f:
            f.write(main)
        with open(os.path.join(test_dir, 'Makefile'), 'w') as f:
            f.write(makefile)
        with open(os.path.join(test_dir, 'data.h'), 'w') as f:
            f.write(header)
        with open(os.path.join(test_dir, 'data.c'), 'w') as f:
            f.write(data)

if __name__ == "__main__":
    main()

