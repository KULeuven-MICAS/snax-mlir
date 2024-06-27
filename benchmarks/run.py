import json
import pathlib
import shutil
import subprocess

try:
    import matplotlib.pyplot as plt

    no_matplotlib = False
except ImportError:
    no_matplotlib = True

# Script that runs a benchmark and processes the results
if __name__ == "__main__":
    benchmark = "tiled_add"
    directory = pathlib.Path("../kernels/tiled_add/")
    benchmark_dir = pathlib.Path.cwd() / "results" / benchmark
    binary = "tiled.acc_dialect.x"
    trace_file = "trace_hart_{hart:05}.trace.json"
    plot_file = binary + ".pdf"

    def announce(string):
        str_len = len(string) + len(benchmark) + 5
        separator = str_len * "="
        print(separator)
        print(" " + string + ' "' + benchmark + '"')
        print(separator)

    def postprocess_trace():
        harts = range(2)
        hart_cycles = {}
        for hart in harts:
            json_file = pathlib.Path(
                benchmark_dir / (binary + ".logs") / trace_file.format(hart=hart)
            )
            with open(json_file) as file:
                cycle_list = []
                json_data = json.load(file)
                for section in json_data:
                    cycle_list.append((section["tstart"], section["tend"]))
            hart_cycles[hart] = cycle_list
        return hart_cycles

    def plot_sections(sections, file):
        # Colors for the sections
        colors = plt.get_cmap("Set2").colors
        fig, ax = plt.subplots()
        # Plotting sections for core 1
        yticks = []
        yticklabels = []
        for j, hart in enumerate(range(len(sections.keys()))):
            yticks.append(15 + j * 10)
            yticklabels.append(f"Hart {j}")
            for i, (start, end) in enumerate(sections[hart]):
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
                    fontsize=8,
                    fontweight="bold",
                    rotation=90,
                )

        # Setting labels and grid
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Time")
        ax.set_title("Performance Sections on SNAX")
        plt.savefig(file, bbox_inches="tight")

    # Copy over files into a new directory
    announce("Preparing benchmark")
    shutil.rmtree(benchmark_dir)
    shutil.copytree(src=directory, dst=benchmark_dir, dirs_exist_ok=False)
    # Run the build
    announce("Building benchmark")
    subprocess.run(["make", binary], cwd=benchmark_dir)
    # Run the code
    announce("Running benchmark")
    subprocess.run(["make", "run_" + binary], cwd=benchmark_dir)
    # Trace the log
    announce("Tracing benchmark")
    subprocess.run(["make", "traces"], cwd=benchmark_dir)
    announce("Postprocessing benchmark")
    hart_sections = postprocess_trace()
    if no_matplotlib:
        announce("Not plotting benchmark, matplotlib not installed")
    else:
        announce("Plotting benchmark")
        plot_sections(hart_sections, plot_file)
