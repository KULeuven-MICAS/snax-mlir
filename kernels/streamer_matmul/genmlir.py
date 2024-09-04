from io import StringIO

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, builtin, func, linalg, transform
from xdsl.ir import Block, Region
from xdsl.printer import Printer

i8 = builtin.IntegerType(builtin.IntAttr(8))


def create_tiled_matrix_multiply(k, m, n, tiling_factors):
    """
    Generate IR in the form of:
    ```
    builtin.module {
      func.func @streamer_matmul(%arg0 : memref<16x16xi8>, %arg1 : memref<16x16xi8,
                                 strided<[1, 16]>>, %arg2 : memref<16x16xi32>) {
        %0 = arith.constant 0 : i32
        linalg.quantized_matmul ins(%arg0, %arg1, %0, %0 : memref<16x16xi8>,
                                    memref<16x16xi8, strided<[1, 16]>>, i32, i32)
                                outs(%arg2 : memref<16x16xi32>)
        func.return
      }
      "transform.sequence"() <{"failure_propagation_mode" = 1 : i32,
                               "operandSegmentSizes" = array<i32: 0, 0>}> ({
      ^0(%arg0 : !transform.any_op, %arg1 : !transform.op<"linalg.quantized_matmul">):
        "transform.yield"() : () -> ()
      }) : () -> ()
    }
    ```
    """

    def get_2d_memref_type(typ, dim_one, dim_two, transpose=False):
        layout = (
            builtin.StridedLayoutAttr([1, dim_one]) if transpose else builtin.NoneAttr()
        )
        return builtin.MemRefType(typ, [dim_one, dim_two], layout=layout)

    input_types = [
        get_2d_memref_type(i8, k, m),
        get_2d_memref_type(i8, m, n, transpose=True),
        get_2d_memref_type(builtin.i32, k, K),
    ]

    b = Block(arg_types=(input_types))

    with ImplicitBuilder(b) as (arg0, arg1, arg2):
        c0 = arith.Constant.from_int_and_width(0, 32)
        linalg.QuantizedMatmulOp([arg0, arg1, c0.result, c0.result], [arg2])
        func.Return()

    region = Region(b)

    function = func.FuncOp.from_region("streamer_matmul", input_types, [], region)

    failurePropagationMode = builtin.IntegerAttr(1, builtin.IntegerType(32))

    input_types_t = [
        transform.AnyOpType(),
        transform.OperationType("linalg.quantized_matmul"),
    ]
    b_t = Block(arg_types=input_types_t)

    with ImplicitBuilder(b_t) as (arg0, arg1):
        (
            tile_op := transform.TileOp(
                arg1, [], tiling_factors, scalable_sizes=tiling_factors
            )
        )
        transform.YieldOp()

    region_t = Region(b_t)

    transform_sequence = transform.SequenceOp(failurePropagationMode, [], [], region_t)

    module = builtin.ModuleOp([function, transform_sequence])

    # Hack for getting mlir-opt-17 compatibility
    tile_op.name = "transform.structured.tile"

    return module


def write_module_to_file(module, file):
    output = StringIO()
    printer = Printer(stream=output)
    printer.print(module)
    with open(file, "w") as output_file:
        output_file.write(output.getvalue())


if __name__ == "__main__":
    M = 16
    N = 16
    K = 16
    tiling_factors = [8, 8]
    module = create_tiled_matrix_multiply(M, K, N, tiling_factors)
    write_module_to_file(module, "test.mlir")
