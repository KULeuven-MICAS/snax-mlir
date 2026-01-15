#map = affine_map<(d0) -> (d0)>
module{
  func.func public @streamer_add(%arg0 : tensor<?xi64>, %arg1 : tensor<?xi64>, %arg2 : tensor<?xi64>) -> tensor<?xi64>{
    %added = linalg.add ins(%arg0, %arg1 : tensor<?xi64>, tensor<?xi64>) outs(%arg2: tensor<?xi64>) -> tensor<?xi64>
    %subbed = linalg.sub ins(%arg0, %added: tensor<?xi64>, tensor<?xi64>) outs(%arg2: tensor<?xi64>) -> tensor<?xi64>
    %mul = linalg.mul ins(%arg0, %subbed: tensor<?xi64>, tensor<?xi64>) outs(%arg2: tensor<?xi64>) -> tensor<?xi64>
    %xori = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %mul : tensor<?xi64>, tensor<?xi64>) outs(%arg2 : tensor<?xi64>) attrs =  {phs_acc = @acc1} {
    ^bb0(%in: i64, %in_0: i64, %out: i64):
      %2 = arith.xori %in, %in_0 : i64
      linalg.yield %2 : i64
    } -> tensor<?xi64>
    return %xori : tensor<?xi64>
  }
}
