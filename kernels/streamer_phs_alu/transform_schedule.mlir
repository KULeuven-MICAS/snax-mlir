module @transforms attributes { transform.with_named_sequence } {
  // Entry point. This takes as the only argument the root operation (typically
  // pass root) given to the transform interpreter.
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
    %matmul = transform.collect_matching @match_generic in %root
      : (!transform.any_op) -> !transform.any_op
    transform.include @annotate_generic failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }

 transform.named_sequence @match_generic(
     %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
   transform.match.operation_name %entry ["linalg.generic"] : !transform.any_op
   transform.match.structured %entry: !transform.any_op {
    ^bb0(%c: !transform.any_op):
      // With 2 inputs.
      %n_ins = transform.match.structured.num_inputs %c
        : (!transform.any_op) -> !transform.param<i64>
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      transform.match.param.cmpi eq %n_ins, %c2 : !transform.param<i64>
   }
   transform.yield %entry : !transform.any_op
 }

  transform.named_sequence @annotate_generic(
    %matmul: !transform.any_op {transform.readonly}) {
      %attr_value = transform.param.constant @acc1 -> !transform.any_param
      transform.annotate %matmul "phs_acc" = %attr_value : !transform.any_op, !transform.any_param
      transform.yield
  }
}
