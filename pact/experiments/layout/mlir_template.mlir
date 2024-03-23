func.func @matmul(%arg0: {memref_a}, %arg1: {memref_b}, %arg2: {memref_c}) {{
    %c0_i32 = arith.constant 0 : i32
    linalg.quantized_matmul ins(%arg0, %arg1, %c0_i32, %c0_i32 : {memref_a}, {memref_b}, i32, i32) outs(%arg2 : {memref_c})
    return
}}
