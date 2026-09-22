#!/usr/bin/env python3
"""Compare a regenerated v3 pool and split with the recorded reference hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_files(folder, expected):
    rows = {}
    for name, digest in expected.items():
        path = folder/name
        actual = None
        if path.is_file():
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        rows[name] = {"matches": actual == digest, "expected": digest, "actual": actual}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT/"datasets/terrestrial_location_v2/campaign")
    parser.add_argument("--split", type=Path, default=ROOT/"datasets/terrestrial_location_v3")
    parser.add_argument("--output", type=Path, default=ROOT/"runs/reference_check.json")
    args = parser.parse_args()
    reference = json.loads((ROOT/"configs/reference_v3_outputs.json").read_text())
    pool = check_files(args.campaign, reference["campaign_sha256"])
    split = check_files(args.split, reference["v3_sha256"])
    passed = all(r["matches"] for r in [*pool.values(), *split.values()])
    report = {"passed": passed, "comparison": "SHA-256 byte equality; provenance files excluded",
              "campaign": pool, "split": split, "model_fitting_or_test_scoring": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"{'PASS' if passed else 'FAIL'}: {len(pool)} pool files and {len(split)} split files; {args.output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
