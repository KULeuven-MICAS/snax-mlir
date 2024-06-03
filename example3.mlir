builtin.module {

    func.func @something(%A: i64, %B: i64, %O: i64, %tiles: index) {
        %c32 = arith.constant 32 : i64

        // set up the loop
        %lb = arith.constant 0 : index
        %step = arith.constant 1 : index

        scf.for %iv = %lb to %tiles step %step {
            // some variables computed in-loop:
            %B_shift = arith.addi %B, %c32 : i64
            %O_shift = arith.addi %O, %c32 : i64

            // launch with loop-invariant and loop-dependent vars:
            %state = accfg.setup "matmul_unit" to ("A" = %A : i64, "B" = %B_shift : i64, "O" = %O_shift : i64, "vector_length" = %c32 : i64) : !accfg.state<"matmul_unit">
            %token = "accfg.launch"(%state) <{"param_names" = [], "accelerator" = "matmul_unit"}> : (!accfg.state<"matmul_unit">) -> !accfg.token<"matmul_unit">
            "accfg.await"(%token) : (!accfg.token<"matmul_unit">) -> ()

            scf.yield
        }

        return
    }
}