from util.tracing.merge_json import merge_json


# All snax compliant configs also use the default rules
module default_rules:
    snakefile:
        "default_rules.smk"
    config:
        config


use rule * from default_rules as default_*


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


rule trace_dasm:
    """
    Use spike-dasm and gen_trace.py to make simulation traces human-readable
    and aggregate stats for a specific hart's trace.
    """
    input:
        "{file}.dasm",
    output:
        temp("{file}_perf.json"),
        temp("{file}.txt"),
    shell:
        "{config[spike-dasm]} < {input} | {config[python]} {config[gen_trace.py]} --permissive -d {output[0]} > {output[1]}"


rule aggregate_json:
    """
    Aggregate traced stats for across chips and hart traces.
    """
    input:
        expand(
            "{file}_trace_chip_{num_chips:02d}_hart_{num_harts:05d}_perf.json",
            file=["{file}"],
            num_chips=range(config["num_chips"]),
            num_harts=range(config["num_harts"]),
        ),
    output:
        temp("{file}_traces.json"),
    run:
        merge_json(input, output[0])


rule perfetto_traces:
    """
    Use trace_to_perfetto to generate visual traces for the simulation.
    """
    input:
        dasm_traces=expand(
            "{file}_trace_chip_{num_chips:02d}_hart_{num_harts:05d}.dasm",
            file=["{file}"],
            num_chips=range(config["num_chips"]),
            num_harts=range(config["num_harts"]),
        ),
        json_traces=expand(
            "{file}_trace_chip_{num_chips:02d}_hart_{num_harts:05d}_perf.json",
            file=["{file}"],
            num_chips=range(config["num_chips"]),
            num_harts=range(config["num_harts"]),
        ),
        elf="{file}.x",
    output:
        "{file}_perfetto_traces.json",
    shell:
        """
        python {config[trace_to_perfetto]} \
            --inputs {input.json_traces} \
            --traces {input.dasm_traces} \
            --elf {input.elf} \
            --accelerator snax_gemmx \
            --output {output}
        """
