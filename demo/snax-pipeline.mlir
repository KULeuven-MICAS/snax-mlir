%0 = memref.alloc() : memref<16x18x18xi8>
%1 = memref.alloc() : memref<16x16x16xi8>
%2 = memref.alloc() : memref<16x16xi8>
%3 = memref.alloc() : memref<10xi8>
%lb = arith.constant 0 : index
%ub = arith.constant 10 : index
%step = arith.constant 1 : index
scf.for %i = %lb to %ub step %step {
  "snax.convolution"(%0, %1) : (memref<16x18x18xi8>, memref<16x16x16xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "snax.maxpool"(%1, %2) : (memref<16x16x16xi8>, memref<16x16xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
  "snax.fully_connected"(%2, %3) : (memref<16x16xi8>, memref<10xi8>) -> ()
  "snax.cluster_sync_op"() : () -> ()
}
