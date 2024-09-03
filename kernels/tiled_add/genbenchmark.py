import pathlib

from benchmark.snax_benchmark import SNAXBenchmark

if __name__ == "__main__":

    def run_all(binary: str, folder: str):
        binary = "tiled.acc_dialect.x"
        folder_no_opt = folder + "_no_opt"
        SIZES = ("ARRAY_SIZE=256", "TILE_SIZE=16", "NO_CHECK=1")

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
        bm.trace()
        bm.process_traces(folder_no_opt)
        bm.copy_binary(folder_no_opt)
        bm.copy_logs(folder_no_opt)

        ## optimised
        folder_opt = folder + "_opt"
        bm = SNAXBenchmark(
            kernel="tiled_add",
            binary=binary,
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
        )
        bm.clean()
        bm.build(build_opts=[*SIZES, "ACCFGOPT=1"])
        bm.run()

        bm.trace()
        bm.process_traces(folder_opt)
        bm.copy_binary(folder_opt)
        bm.copy_logs(folder_opt)

    binaries = {
        "run0": "untiled.acc_dialect.x",
        "run1": "tiled.acc_dialect.x",
        "run2": "tiled_pipelined.acc_dialect.x",
    }

    for folder, binary in binaries.items():
        run_all(binary, folder)
