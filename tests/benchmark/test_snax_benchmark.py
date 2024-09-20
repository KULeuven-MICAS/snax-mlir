import pathlib

from util.snax_benchmark import SNAXBenchmark


def test_snax_benchmark_runner():
    folder = "test_run"
    this_file = pathlib.Path(__file__)
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary="untiled.acc_dialect.x",
        src_dir=str(this_file.parent / ".." / ".." / "kernels" / "tiled_add" / ""),
        export_dir=str(this_file.parent),
    )
    bm.clean()
    bm.build(build_opts=["ARRAY_SIZE=128", "TILE_SIZE=16", "NO_CHECK=1"])
    bm.run()
    bm.trace()
    bm.process_traces(folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
