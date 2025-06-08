#!/usr/bin/env python3
"""Extract trajectory data from all available runs."""

import json
from pathlib import Path
from extract_run_data import extract_and_save_trajectory


def extract_all_runs():
    """Extract trajectory data from all runs in the index."""

    # Load runs index
    with open("visualization/3d/runs_index.json", "r") as f:
        index = json.load(f)

    runs = index["runs"]
    print(f"Found {len(runs)} runs to extract")

    extracted = 0
    for i, run in enumerate(runs):
        print(f"\n[{i + 1}/{len(runs)}] Processing: {run['label']}")

        h5_path = run["path"]
        run_dir = Path(h5_path).parent.name
        output_filename = f"trajectory_{run_dir}.json"

        # Check if already extracted
        output_path = Path("visualization/3d") / output_filename
        if output_path.exists():
            print(f"  Already extracted: {output_filename}")
            extracted += 1
            continue

        # Extract
        try:
            if extract_and_save_trajectory(h5_path, output_filename):
                extracted += 1
            else:
                print(f"  Failed to extract")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nExtraction complete: {extracted}/{len(runs)} runs processed")


if __name__ == "__main__":
    extract_all_runs()
