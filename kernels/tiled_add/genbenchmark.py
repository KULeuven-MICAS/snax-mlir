import pathlib

from benchmark.snax_benchmark import SNAXBenchmark


def generate_traces(folder: str, binary: str):
    """
    make traces with Markus beautiful tool
    """
    import subprocess
    traces = [
        f'results/tiled_add/{folder}/trace_hart_{i:05}.dasm' for i in range(2)
    ]
    logs = [
        f'results/tiled_add/{folder}/trace_hart_{i:05}.trace.json' for i in range(2)
    ]
    subprocess.check_output(
        ["python3", "../../benchmark/trace_to_perfetto.py",
         '--traces', *traces,
         '-i', *logs,
         '--elf', f'results/tiled_add/{folder}/{binary}',
         '--addr2line', 'llvm-addr2line',
         '-o', f'{folder}_events.json'
        ]
    )


if __name__ == "__main__":
    binary = "tiled.acc_dialect.x"
    folder = "no_opt"
    SIZES = ("ARRAY_SIZE=256", "TILE_SIZE=16", 'NO_CHECK=1')
    ## not optimised
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
    )
    bm.clean()
    bm.build(build_opts=[*SIZES])
    bm.run()
    hart_cycles = bm.trace()
    bm.plot(hart_cycles, folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
    generate_traces(folder, binary)
    ## optimised
    folder = "opt"
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
    )
    bm.clean()
    bm.build(build_opts=[*SIZES, "ACCFGOPT=1"])
    bm.run(['-i'])
    hart_cycles = bm.trace()
    bm.plot(hart_cycles, folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
    generate_traces(folder, binary)
