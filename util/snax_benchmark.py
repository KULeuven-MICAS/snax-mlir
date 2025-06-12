import pathlib
import shutil
import subprocess
from collections.abc import Callable

from util.tracing.trace_to_perfetto import process_traces


class SNAXBenchmark:
    input_file = "trace_chip_00_hart_{hart}.trace.json"
    trace_file = "trace_chip_00_hart_{hart}.dasm"

    def __init__(
        self,
        kernel: str,
        binary: str,
        src_dir: str,
        # export dir: for all results (useful for manual inspection)
        export_dir: str,
        # output dir: for all benchmark outputs (for docs generation)
        output_dir: str,
        benchmark: str | None = None,
    ):
        self.kernel = kernel
        self.binary = binary
        self.src_dir = pathlib.Path(src_dir)
        self.log_dir = self.src_dir / (self.binary + ".logs")
        if benchmark is None:
            self.benchmark = kernel
        else:
            self.benchmark = benchmark
        self.export_dir = export_dir / pathlib.Path("results") / self.benchmark
        self.output_dir = output_dir / pathlib.Path("output") / self.benchmark

    def announce(self, string) -> None:
        str_len = len(string) + len(self.benchmark) + 5
        separator = str_len * "="
        print(separator)
        print(" " + string + ' "' + self.benchmark + '"')
        print(separator)

    def build(self, build_opts: list[str] = []) -> None:
        self.announce("Building benchmark")
        subprocess.run(["make", self.binary, *build_opts], cwd=self.src_dir, check=True)

    def clean(self) -> None:
        self.announce("Cleaning benchmark")
        subprocess.run(["make", "clean"], cwd=self.src_dir, check=True)

    def run(self) -> None:
        self.announce("Running benchmark")
        subprocess.run(["make", "run_" + self.binary], cwd=self.src_dir, check=True)

    def trace(self):
        self.announce("Tracing benchmark")
        subprocess.run(["make", "traces"], cwd=self.src_dir, check=True)

    def plot(self):
        self.announce("Generating plots")
        subprocess.run(["make", "plots"], cwd=self.src_dir, check=True)

    def process_traces(self, folder: str, accelerator: str | None = None, file=None):
        self.announce("Processing Traces")
        dst_folder = self.export_dir / pathlib.Path(folder)
        if file is None:
            file = dst_folder / (self.binary + "_events.json")
        input_filenames = self.log_dir.glob(__class__.input_file.format(hart="*"))
        inputs = [open(input_file, "rb") for input_file in input_filenames]
        trace_filenames = [str(path) for path in self.log_dir.glob(__class__.trace_file.format(hart="*"))]

        if not dst_folder.exists():
            dst_folder.mkdir(parents=True)
        output_events = open(dst_folder / file, "w")
        process_traces(
            inputs,
            trace_filenames,
            str(self.src_dir / self.binary),
            accelerator=accelerator,
            addr2line="llvm-addr2line",
            output=output_events,
        )
        output_events.close()

    def generate_output_log(self, generator: Callable[[str], str]) -> None:
        self.announce("Generating output log")
        output_folder = pathlib.Path(self.output_dir)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        with open(output_folder / "index.md", "w") as f:
            f.write(generator(self.benchmark))

    def copy_binary(self, folder: str):
        self.announce("Copying binary")
        dst_folder = pathlib.Path(self.export_dir / folder)
        if not dst_folder.exists():
            dst_folder.mkdir(parents=True)
        shutil.copy(src=self.src_dir / self.binary, dst=dst_folder / self.binary)

    def copy_logs(self, folder: str):
        self.announce("Copying logs")
        shutil.copytree(src=self.log_dir, dst=self.export_dir / folder, dirs_exist_ok=True)

    def copy_plots(self):
        self.announce("Copying plots to output folder")
        plot_filenames = tuple(self.src_dir.glob("*.png"))
        plot_filenames += tuple(self.src_dir.glob("*.pdf"))
        output_folder = pathlib.Path(self.output_dir) / "figures"
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        for plot in plot_filenames:
            shutil.copy(src=plot, dst=output_folder)

    def copy_results(self):
        self.announce("Copying results to output folder")
        archive_path = shutil.make_archive(self.benchmark + "_results", "gztar", self.export_dir)
        output_folder = pathlib.Path(self.output_dir)
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        shutil.move(archive_path, self.output_dir)
