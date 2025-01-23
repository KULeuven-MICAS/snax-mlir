from util.snake.paths import get_traces


rule simulate:
    input:
        "{file}.x",
    output:
        *[
            temp(trace)
            for trace in get_traces(
                ["{file}"], config["num_chips"], config["num_harts"], "dasm"
            )
        ],
    log:
        "{file}.vltlog",
    shell:
        "{config[vltsim]} --prefix-trace={wildcards.file}_ {wildcards.file}.x  2>&1 | tee {log}"
