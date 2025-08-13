"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x64xi32>) -> tensor<1x64xi32>, sym_name = "softmax", sym_visibility = "public"}> ({
  ^bb0(%arg0: tensor<1x64xi32>):
    %0 = "tensor.empty"() : () -> tensor<1x64xi32>
    %1 = "linalg.softmax"(%arg0, %0) <{dimension = 1 : i64}> : (tensor<1x64xi32>, tensor<1x64xi32>) -> tensor<1x64xi32>
    "func.return"(%1) : (tensor<1x64xi32>) -> ()
  }) : () -> ()
}) : () -> ()