rule simulate:
    input:
        "{file}.x",
    output:
        temp(
            expand(
                "{file}_trace_chip_{num_chips:02d}_hart_{num_harts:05d}.dasm",
                file=["{file}"],
                num_chips=range(config["num_chips"]),
                num_harts=range(config["num_harts"]),
            ),
        ),
    log:
        "{file}.vltlog",
    shell:
        "{config[vltsim]} --prefix-trace={wildcards.file}_ {wildcards.file}.x  2>&1 | tee {log}"
