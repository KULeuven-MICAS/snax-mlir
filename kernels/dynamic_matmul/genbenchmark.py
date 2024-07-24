import pathlib

from benchmark.snax_benchmark import SNAXBenchmark

if __name__ == "__main__":

    def run_all(binary: str, folder: str):
        bm = SNAXBenchmark(
            kernel="dynamic_matmul",
            binary=binary,
            src_dir=str(pathlib.Path.cwd()),
            export_dir=str(pathlib.Path.cwd()),
        )
        bm.clean()
        bm.build(build_opts=[])
        bm.run()
        hart_cycles = bm.trace()
        bm.plot(hart_cycles, folder)
        bm.copy_binary(folder)
        bm.copy_logs(folder)

    binaries = {"run0": "dynamic_matmul.x"}

    for folder, binary in binaries.items():
        run_all(binary, folder)
