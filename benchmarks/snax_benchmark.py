import json
import pathlib
import shutil
import subprocess


class SNAXBenchmark:
    trace_file = "trace_hart_{hart}.trace.json"

    def __init__(
        self, kernel: str, binary: str, export_dir: str, benchmark: str | None = None
    ):
        self.kernel = kernel
        self.binary = binary
        self.src_dir = pathlib.Path(f"../kernels/{self.kernel}/")
        self.log_dir = self.src_dir / (self.binary + ".logs")
        if benchmark is None:
            self.benchmark = kernel
        else:
            self.benchmark = benchmark
        self.export_dir = export_dir / pathlib.Path("results") / self.benchmark

    def announce(self, string) -> None:
        str_len = len(string) + len(self.benchmark) + 5
        separator = str_len * "="
        print(separator)
        print(" " + string + ' "' + self.benchmark + '"')
        print(separator)

    def build(self, build_opts: list[str] = []) -> None:
        self.announce("Building benchmark")
        print(["make", self.binary, *build_opts])
        subprocess.run(["make", self.binary, *build_opts], cwd=self.src_dir, check=True)

    def run(self) -> None:
        self.announce("Running benchmark")
        subprocess.run(["make", "run_" + self.binary], cwd=self.src_dir, check=True)

    def trace(self) -> dict[int, list[tuple[int, int]]]:
        self.announce("Tracing benchmark")
        subprocess.run(["make", "traces"], cwd=self.src_dir, check=True)
        # Get number of harts
        harts = len(list(self.log_dir.glob(__class__.trace_file.format(hart="*"))))
        hart_cycles = {}
        for hart in range(harts):
            json_file = pathlib.Path(
                self.log_dir / __class__.trace_file.format(hart=f"{hart:05}")
            )
            with open(json_file) as file:
                cycle_list = []
                json_data = json.load(file)
                for section in json_data:
                    cycle_list.append((section["tstart"], section["tend"]))
            hart_cycles[hart] = cycle_list
        return hart_cycles

    def plot(
        self, hart_cycles: dict[int, list[tuple[int, int]]], folder: str, file=None
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        dst_folder = self.export_dir / pathlib.Path(folder)
        if file is None:
            file = dst_folder / (self.binary + ".pdf")
        self.announce(f"Plotting benchmark @ {file}")
        # Colors for the sections
        colors = plt.get_cmap("Set2").colors
        _, ax = plt.subplots(figsize=(30, 6))
        # Plotting sections for core 1
        yticks = []
        yticklabels = []
        for j, hart in enumerate(range(len(hart_cycles.keys()))):
            yticks.append(15 + j * 10)
            yticklabels.append(f"Hart {j}")
            for i, (start, end) in enumerate(hart_cycles[hart]):
                ax.broken_barh(
                    [(start, end - start)],
                    ((j + 1) * 10, 9),
                    facecolors=colors[i % len(colors)],
                    label=f"Core 1 Section {i+1}",
                )
                ax.text(
                    start + (end - start) / 2,
                    (j + 1) * 10 + 9 / 2,
                    f"{int(end - start)}",
                    va="center",
                    ha="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                    rotation=90,
                )
        # Setting labels and grid
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title("Performance Sections on SNAX")
        if not dst_folder.exists():
            dst_folder.mkdir(parents=True)
        plt.savefig(dst_folder / file, bbox_inches="tight")

    def copy_binary(self, folder: str):
        self.announce("Copying binary")
        dst_folder = pathlib.Path(self.export_dir / folder)
        if not dst_folder.exists():
            dst_folder.mkdir()
        shutil.copy(src=self.src_dir / self.binary, dst=dst_folder / self.binary)

    def copy_logs(self, folder: str):
        self.announce("Copying logs")
        shutil.copytree(
            src=self.log_dir, dst=self.export_dir / folder, dirs_exist_ok=True
        )


if __name__ == "__main__":
    folder = "run1"
    bm = SNAXBenchmark(
        kernel="tiled_add",
        binary="tiled.acc_dialect.x",
        export_dir=str(pathlib.Path.cwd()),
    )
    bm.build(build_opts=["ARRAY_SIZE=32768", "TILE_SIZE=1024"])
    bm.run()
    hart_cycles = bm.trace()
    bm.plot(hart_cycles, folder)
    bm.copy_binary(folder)
    bm.copy_logs(folder)
