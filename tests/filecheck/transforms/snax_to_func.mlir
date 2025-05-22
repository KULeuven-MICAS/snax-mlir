// RUN: snax-opt --split-input-file %s -p snax-to-func --print-op-generic | filecheck %s

"builtin.module"() ({
  "snax.cluster_sync_op"() : () -> ()
}) : () -> ()

//CHECK: "builtin.module"() ({
//CHECK-NEXT:   "func.call"() <{callee = @snax_cluster_hw_barrier}> : () -> ()
//CHECK-NEXT:   "func.func"() <{sym_name = "snax_cluster_hw_barrier", function_type = () -> (), sym_visibility = "private"}> ({
//CHECK-NEXT:   }) : () -> ()
//CHECK-NEXT: }) : () -> ()
