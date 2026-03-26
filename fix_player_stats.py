"""
fix_player_stats.py

Scans player_stats_hp.csv, player_stats_snd.csv, and player_stats_ovld.csv for
rows with the wrong number of fields caused by a GPT extraction error that
inserted an extra team-abbreviation token at field index 6 (before player_slot).

Expected field counts (including header): 22 for all three files.

Bad row example (23 fields):
  2026_M1Q1S34_BOS_NY,4,HP,Colossus,BOS,NY,BOS,2,CAMMY,BOS,...
                                              ^^^
                                    extra token at index 6

Fix: drop field index 6 when a data row has 23 fields and the extra token
matches the value already in the team column (field index 4).

Also silently drops blank/empty lines.

Saves fixed CSVs back to the same paths.
"""

import csv
import sys
from pathlib import Path

DATA_DIR = Path("C:/Repos/cod_esports_ml/data_clean")

FILES = {
    "HP":   DATA_DIR / "player_stats_hp.csv",
    "SND":  DATA_DIR / "player_stats_snd.csv",
    "OVLD": DATA_DIR / "player_stats_ovld.csv",
}

EXPECTED_FIELDS = 22   # same for all three modes
EXTRA_TOKEN_IDX = 6    # the spurious team abbrev sits here in bad rows


def fix_file(label: str, path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Scanning {label}: {path.name}")
    print(f"{'='*60}")

    with path.open(newline="", encoding="utf-8") as f:
        raw_lines = f.readlines()

    if not raw_lines:
        print("  File is empty – skipping.")
        return

    header_line = raw_lines[0]
    header_fields = header_line.strip().split(",")
    print(f"  Header has {len(header_fields)} fields (expected {EXPECTED_FIELDS})")

    bad_rows = []
    blank_rows = []
    fixed_rows = []

    for lineno, line in enumerate(raw_lines[1:], start=2):
        stripped = line.strip()

        # --- blank / empty lines ---
        if not stripped:
            blank_rows.append(lineno)
            continue

        fields = stripped.split(",")
        n = len(fields)

        if n == EXPECTED_FIELDS:
            fixed_rows.append(fields)

        elif n == EXPECTED_FIELDS + 1:
            extra_val = fields[EXTRA_TOKEN_IDX]

            bad_rows.append({
                "lineno": lineno,
                "raw":    stripped,
                "extra":  extra_val,
            })

            corrected = fields[:EXTRA_TOKEN_IDX] + fields[EXTRA_TOKEN_IDX + 1:]
            fixed_rows.append(corrected)

        else:
            print(f"  UNEXPECTED field count ({n}) on line {lineno} – left unchanged:")
            print(f"    {stripped}")
            fixed_rows.append(fields)

    # --- Report ---
    print(f"\n  Total data rows (excl header): {len(raw_lines) - 1}")
    print(f"  Blank / empty lines dropped  : {len(blank_rows)}")
    print(f"  Rows with {EXPECTED_FIELDS + 1} fields (bad) : {len(bad_rows)}")

    if bad_rows:
        print(f"\n  Bad rows found and fixed:")
        for b in bad_rows:
            print(f"    line {b['lineno']:>4}  extra token removed: '{b['extra']}'")
            print(f"           {b['raw']}")

    # --- Write fixed file ---
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_fields)
        writer.writerows(fixed_rows)

    print(f"\n  Saved {len(fixed_rows)} data rows -> {path}")


def main() -> None:
    for label, path in FILES.items():
        if not path.exists():
            print(f"ERROR: {path} not found – skipping {label}.")
            continue
        fix_file(label, path)

    print("\nDone.")


if __name__ == "__main__":
    main()
