"""Validate a completed run and optionally compare spectra with supplied results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(out, reference=None):
    config = json.loads((out / "simulation_config.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    require(manifest["exit_code"] == 0, "Simulator did not finish successfully")
    duration = config["backend"]["duration_s"]
    nt = round(duration / config["backend"]["time_resolution_s"])
    report = dict(mode=config["transmitter_type"], elapsed_seconds=manifest["elapsed_seconds"], receivers={})
    for rx in config["receivers"]:
        name = rx["name"].lower()
        filename = f"{name}_synthetic_{duration}s_v3_2.npz"
        with np.load(out / filename, allow_pickle=False) as data:
            spec = data["avg_specgram_dbm"]
            freqs = data["freq_mhz"]
            require(spec.shape == (nt, len(freqs)), f"{name}: wrong spectrum shape")
            require(np.isfinite(spec).all(), f"{name}: nonfinite spectrum")
            require(np.all(np.diff(freqs) > 0), f"{name}: frequency ordering")
            require(np.allclose(np.diff(data["time_s_rel"]), config["backend"]["time_resolution_s"]), f"{name}: time step")
            require(float(data["receiver_lat"]) == rx["lat_deg"], f"{name}: latitude")
            states = json.loads(str(data["lte_states_json"]))
            require(set(states) == {"idle", "light", "medium", "loaded"}, f"{name}: missing genuine IQ state")
            counts = {k: v["template_count"] for k, v in states.items()}
            incidents = json.loads((out / f"{name}_emily_candidate_incidents.json").read_text())
            for row in incidents:
                require(row["metadata"]["synthetic"] is True, "Missing synthetic marker")
                require(row["lat"] == rx["lat_deg"] and row["lon"] == rx["lon_deg"], "Incident coordinates must identify receiver")
                start, end = row["t_start_offset_sec"], row["t_end_offset_sec"]
                # Open-ended incidents deliberately use null for unobserved boundaries.
                require(start is None or 0 <= start < duration, "Incident start range")
                require(end is None or 0 <= end < duration, "Incident end range")
                require(start is None or end is None or start <= end, "Incident time order")
                require(end is not None or row["ongoing"], "Unexplained missing end")
                require(row["f_low_hz"] <= row["f_high_hz"], "Incident frequency range")
            result = dict(shape=list(spec.shape), finite=True, templates_per_state=counts,
                          incident_count=len(incidents), incident_intervals_s=[
                              [r["t_start_offset_sec"], r["t_end_offset_sec"]] for r in incidents])
            if config["transmitter_type"] == "leo":
                for key in ("satellite_el_deg_t", "slant_range_km_t", "applied_doppler_hz_t"):
                    require(data[key].shape == (nt,) and np.isfinite(data[key]).all(), f"Invalid {key}")
                result["visible_seconds"] = int(data["satellite_visible_t"].sum())
                result["max_elevation_deg"] = float(data["satellite_el_deg_t"].max())
            if reference:
                with np.load(reference / filename, allow_pickle=False) as original:
                    require(spec.shape == original["avg_specgram_dbm"].shape, "Reference shape mismatch")
                    diff = np.abs(spec - original["avg_specgram_dbm"])
                    result["reference_spectrum_abs_diff_db"] = dict(mean=float(diff.mean()), maximum=float(diff.max()))
                original_incidents = json.loads((reference / f"{name}_emily_candidate_incidents.json").read_text())
                result["reference_incident_count"] = len(original_incidents)
                result["reference_incident_intervals_s"] = [[r["t_start_offset_sec"], r["t_end_offset_sec"]] for r in original_incidents]
            report["receivers"][name] = result
    plots = sorted((out / "plots").glob("*.png"))
    expected = 13 if config["transmitter_type"] == "leo" else 12
    require(len(plots) == expected, f"Expected {expected} plots, got {len(plots)}")
    for path in plots:
        with Image.open(path) as img:
            img.verify()
    report["verified_png_count"] = len(plots)
    report["link_budget"] = pd.read_csv(out / "ground_truth_link_budget.csv").to_dict("records")
    if config["transmitter_type"] == "leo":
        links = pd.read_csv(out / "ground_truth_leo_links.csv")
        require(len(links) == nt * len(config["receivers"]), "Missing per-second LEO truth")
        report["leo_truth_rows"] = len(links)
    (out / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    validate(args.output_dir, args.reference)
