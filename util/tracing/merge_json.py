import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence


def merge_json(file_list: Sequence[str], output_file: str):
    prefix = ""
    traces: defaultdict[
        int, defaultdict[int, list[dict[str, int | float | str | None]]]
    ] = defaultdict(lambda: defaultdict(list))
    for filename in file_list:
        match = re.match(r"(.*)_trace_chip_(\d{2})_hart_(\d{5})_perf\.json", filename)
        if match:
            prefix = match.group(1)
            chip_id = int(match.group(2))
            hart_id = int(match.group(3))

            # Load JSON content
            with open(filename) as f:
                fragments = json.load(f)

            # Append the fragments to the corresponding chip and hart
            traces[chip_id][hart_id].extend(fragments)
        else:
            raise RuntimeError(
                f"Filename '{filename}' doesn't match the expected regex"
            )

    # Convert the defaultdict to a nested list structure
    result = [
        [chip_harts[hart_id] for hart_id in sorted(chip_harts)]
        for _, chip_harts in sorted(traces.items())
    ]

    # Use prefix for output file if not explicitly provided
    output_file = output_file or f"{prefix}_aggregated.json"

    # Output the aggregated result
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge JSON files containing chip and hart traces."
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        nargs="+",
        help="List of JSON files to merge."
        "Filenames must match the format"
        "'<prefix>_trace_chip_<chip_id>_hart_<hart_id>_perf.json'.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT_FILE",
        default=None,
        help="Name of the output JSON file."
        "Defaults to '<prefix>_aggregated.json' based on input filenames.",
    )

    args = parser.parse_args()
    merge_json(args.files, args.output)
