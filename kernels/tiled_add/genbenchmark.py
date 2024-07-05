import pathlib

from benchmark.snax_benchmark import SNAXBenchmark

if __name__ == "__main__":
    # def run_all(binary: str, folder: str):
    #     bm = SNAXBenchmark(
    #         kernel="tiled_add",
    #         binary=binary,
    #         src_dir=str(pathlib.Path.cwd()),
    #         export_dir=str(pathlib.Path.cwd()),
    #     )
    #     bm.clean()
    #     bm.build(build_opts=["ARRAY_SIZE=256", "TILE_SIZE=16", "NO_CHECK=1"])
    #     bm.run()
    #     hart_cycles = bm.trace()
    #     bm.plot(hart_cycles, folder)
    #     bm.copy_binary(folder)
    #     bm.copy_logs(folder)

    # binaries = {
    #     "run0": "untiled.acc_dialect.x",
    #     "run1": "tiled.acc_dialect.x",
    #     "run2": "tiled_pipelined.acc_dialect.x",
    # }

    # for folder, binary in binaries.items():
    #     run_all(binary, folder)

    binary = "tiled.acc_dialect.x"
    folder = "no_opt"
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
    )
    bm.clean()
    bm.build(build_opts=["ARRAY_SIZE=256", "TILE_SIZE=16"])
    bm.run()
    hart_cycles = bm.trace()
    bm.plot(hart_cycles, folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
    folder = "opt"
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary=binary,
        src_dir=str(pathlib.Path.cwd()),
        export_dir=str(pathlib.Path.cwd()),
    )
    bm.clean()
    bm.build(build_opts=["ARRAY_SIZE=256", "TILE_SIZE=16", "ACCFGOPT=1"])
    bm.run(['-i'])
    hart_cycles = bm.trace()
    bm.plot(hart_cycles, folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
