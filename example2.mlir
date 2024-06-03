builtin.module {

    "accfg.accelerator"() <{
        name               = @matmul_unit,
        fields             = {A=0x3c0, B=0x3c1, O=0x3c2, size=0x3c3},
        launch_fields      = {},
        barrier            = 0x7c3
    }> : () -> ()

    func.func @monster(%A: i64, %B: i64, %O: i64, %size: i64) {
        %state = accfg.setup "matmul_unit" to ("A" = %A : i64, "B" = %B : i64, "O" = %O : i64, "size" = %size : i64) : !accfg.state<"matmul_unit">
        %token = "accfg.launch"(%state) <{param_names = [], accelerator = "matmul_unit"}> : (!accfg.state<"matmul_unit">) -> !accfg.token<"matmul_unit">
        "accfg.await"(%token) : (!accfg.token<"matmul_unit">) -> ()

        %state2 = accfg.setup "matmul_unit" to ("A" = %A : i64, "O" = %B : i64, "B" = %O : i64, "size" = %size : i64) : !accfg.state<"matmul_unit">
        %token2 = "accfg.launch"(%state2) <{param_names = [], accelerator = "matmul_unit"}> : (!accfg.state<"matmul_unit">) -> !accfg.token<"matmul_unit">
        "accfg.await"(%token2) : (!accfg.token<"matmul_unit">) -> ()

        return
    }
}
