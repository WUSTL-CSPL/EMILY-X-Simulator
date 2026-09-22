#!/usr/bin/env python3
"""Check only the IQ files needed for the selected workflow; optionally pin v3 inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_inputs(mode="dataset", reference_v3=False):
    source = ROOT/"Incident_simulator/emilyx_lte_end_to_end_simulator_v3_2.py"
    if not source.is_file():
        raise ValueError("Missing simulator. Run: git submodule update --init --recursive")
    states = ("loaded",) if mode == "dataset" else ("idle", "light", "medium", "loaded")
    profiles = [("srsran_d2c", 5_760_000)] if mode == "leo" else [("srsran", 11_520_000)]
    if mode == "all":
        profiles.append(("srsran_d2c", 5_760_000))
    rows = []
    for prefix, rate in profiles:
        for state in states:
            path = ROOT/"EMILY-X"/f"{prefix}_{state}.cf32"
            metadata = path.with_suffix(".cf32.json")
            if not path.is_file() or not metadata.is_file():
                raise ValueError(f"Provide {path.name} and its .json sidecar in EMILY-X/")
            meta = json.loads(metadata.read_text())
            if path.stat().st_size != meta["expected_bytes"] or meta["sample_rate_hz"] != rate or meta["dtype"] != "complex64":
                raise ValueError(f"IQ size or radio profile mismatch: {path.name}")
            iq = np.memmap(path, dtype=np.complex64, mode="r")
            if len(iq) != meta["expected_samples"] or not np.isfinite(iq).all():
                raise ValueError(f"Invalid IQ samples: {path.name}")
            rms = float(np.sqrt(np.mean(np.abs(iq)**2)))
            if rms <= 0:
                raise ValueError(f"Silent IQ: {path.name}")
            rows.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path),
                         "bytes": path.stat().st_size, "samples": len(iq), "rms": rms})
            print(f"OK {path.name}: {len(iq)} samples")
    if mode == "dataset" and not any((ROOT/"srtm_data").rglob("*.hgt")):
        raise ValueError("Dataset generation needs cached SRTM .hgt tiles in srtm_data/; see docs/DATA.md")
    if reference_v3:
        reference = json.loads((ROOT/"configs/reference_v3_inputs.json").read_text())
        commit = subprocess.check_output(["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
        if commit != reference["simulator_commit"] or digest(source) != reference["simulator_sha256"]:
            raise ValueError("Simulator differs from the v3 reference")
        for directory, entries in (("EMILY-X", reference["iq_sha256"]), ("srtm_data", reference["terrain_sha256"])):
            for name, expected in entries.items():
                path = ROOT/directory/name
                if not path.is_file() or digest(path) != expected:
                    raise ValueError(f"Missing or different reference input: {directory}/{name}")
        print("OK pinned simulator, loaded IQ and all reference terrain tiles")
    (ROOT/"input_manifest.json").write_text(json.dumps(rows, indent=2)+"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dataset", "terrestrial", "leo", "all"), default="dataset")
    parser.add_argument("--reference-v3", action="store_true")
    args = parser.parse_args()
    try:
        check_inputs(args.mode, args.reference_v3)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
