#!/usr/bin/env python3
"""Replay the fixed v3 split with configurable directional receiving antennas.

The original generator and upstream simulator are imported, never edited.
Paths in the JSON config are relative to this repository, not the shell cwd.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
from dataclasses import asdict, replace
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile

import generate_location_dataset as base

np = base.np
ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
TABLES = ("inputs.jsonl", "labels.jsonl", "ground_truth.jsonl")
SCENARIO_FIELDS = ("sample_id", "x_m", "y_m", "latitude_deg", "longitude_deg",
                   "cell_x", "cell_y", "receiver_seeds", "geometry_replacements")
PATTERNS = {}
BASE_RECEIVERS = ()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def gain_dbi(off_axis_deg, pattern):
    """HPBW is the full width: +/- HPBW/2 is -3 dB from peak."""
    angle = (float(off_axis_deg) + 180.0) % 360.0 - 180.0
    return pattern["peak_gain_dbi"] - min(
        12.0 * (angle / pattern["horizontal_hpbw_deg"]) ** 2,
        pattern["max_attenuation_db"])


def validate_pattern(pattern):
    expected = {"model", "pointing", "horizontal_hpbw_deg", "peak_gain_dbi",
                "max_attenuation_db", "vertical_pattern"}
    if set(pattern) != expected:
        raise ValueError(f"Pattern fields must be {sorted(expected)}")
    if (pattern["model"] != "horizontal_parabolic" or
            pattern["pointing"] != "fixed_toward_baseline_region_center" or
            pattern["vertical_pattern"] != "none"):
        raise ValueError("Only a fixed horizontal parabolic pattern is supported")
    width, peak, attenuation = [pattern[k] for k in
        ("horizontal_hpbw_deg", "peak_gain_dbi", "max_attenuation_db")]
    if not all(math.isfinite(v) for v in (width, peak, attenuation)):
        raise ValueError("Pattern values must be finite")
    if not 0 < width <= 360 or peak != 0 or attenuation < 0:
        raise ValueError("Use 0 < HPBW <= 360, peak gain 0 dBi and attenuation >= 0")


def resolve_patterns(source_config, pattern):
    validate_pattern(pattern)
    patterns = {}
    for rx in source_config["receivers"]:
        bearing, _, _ = base.sim.GEOD.inv(
            rx["lon_deg"], rx["lat_deg"], source_config["center_longitude_deg"],
            source_config["center_latitude_deg"])
        patterns[rx["name"]] = {**pattern, "azimuth_deg": bearing % 360.0}
    return patterns


def evaluate_pattern(rx, scenario, pattern):
    bearing, _, _ = base.sim.GEOD.inv(rx.lon_deg, rx.lat_deg,
        scenario["longitude_deg"], scenario["latitude_deg"])
    off_axis = (bearing - pattern["azimuth_deg"] + 180.0) % 360.0 - 180.0
    return {"bearing_rx_to_tx_deg": bearing % 360.0,
            "off_axis_deg": off_axis, "effective_gain_dbi": gain_dbi(off_axis, pattern)}


def initialize_worker(freq, models, config, patterns):
    global PATTERNS, BASE_RECEIVERS
    base.initialize_worker(freq, models, config)
    BASE_RECEIVERS = tuple(replace(rx) for rx in base.sim.RECEIVERS)
    PATTERNS = patterns


def simulate_directional(scenario):
    evaluations = {rx.name: evaluate_pattern(rx, scenario, PATTERNS[rx.name])
                   for rx in BASE_RECEIVERS}
    # Source and RX are stationary. One effective gain per link is therefore
    # sufficient. The simulator applies it to source power BEFORE adding noise
    # and running the detector; we never attenuate an already noisy spectrum.
    base.sim.RECEIVERS = [replace(rx, antenna_gain_dbi=
        evaluations[rx.name]["effective_gain_dbi"]) for rx in BASE_RECEIVERS]
    try:
        row = base.simulate_scene(scenario)
        base.validate_row(row)
    finally:
        base.sim.RECEIVERS = list(BASE_RECEIVERS)
    # Keep fixed hardware metadata separate from source-dependent effective gain.
    row["truth"]["receivers"] = [asdict(rx) for rx in BASE_RECEIVERS]
    row["truth"]["receiver_antenna_patterns"] = PATTERNS
    for name, evaluation in evaluations.items():
        row["truth"]["links"][name]["rx_antenna_response"] = evaluation
    return row


def load_baseline(source, source_config):
    result, seen = {}, set()
    expected_rxs = source_config["receivers"]
    expected_tx = asdict(replace(base.ORIGINAL_TX,
        height_agl_m=source_config["profile"]["tx_height_m"],
        electrical_downtilt_deg=source_config["profile"]["downtilt_deg"]))
    for split in SPLITS:
        tables = {name: read_jsonl(source / split / name) for name in TABLES}
        ids = [row["sample_id"] for row in tables["inputs.jsonl"]]
        for name in TABLES:
            if [row["sample_id"] for row in tables[name]] != ids:
                raise ValueError(f"Unaligned baseline tables in {split}")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Duplicate sample IDs within or across splits")
        seen.update(ids)
        for label, truth in zip(tables["labels.jsonl"], tables["ground_truth.jsonl"]):
            expected = {**expected_tx, "lat_deg": truth["latitude_deg"],
                        "lon_deg": truth["longitude_deg"]}
            if truth["transmitter"] != expected or truth["receivers"] != expected_rxs:
                raise ValueError("Baseline hardware differs from this replay's simulator")
            if truth["profile"] != source_config["profile"] or not truth["source_present"]:
                raise ValueError("Unsupported baseline profile or source-off scene")
            if label != {"sample_id": truth["sample_id"], "transmitter_type": "terrestrial_lte",
                         "latitude_deg": truth["latitude_deg"], "longitude_deg": truth["longitude_deg"]}:
                raise ValueError("Labels do not match baseline source truth")
        result[split] = tables
    return result


def flags(row):
    return tuple(bool(rx["incidents"]) for rx in row["receivers"])


def coverage(rows):
    counts = {"both": 0, "RX1_only": 0, "RX2_only": 0, "neither": 0}
    names = {(True, True): "both", (True, False): "RX1_only",
             (False, True): "RX2_only", (False, False): "neither"}
    for row in rows:
        counts[names[flags(row)]] += 1
    return {"samples": len(rows), **counts,
            "any_detection_fraction": 1.0 - counts["neither"] / len(rows),
            "both_detection_fraction": counts["both"] / len(rows),
            "incidents": sum(len(rx["incidents"]) for row in rows for rx in row["receivers"])}


def verify_physics(row, old_truth):
    truth = row["truth"]
    for key in SCENARIO_FIELDS + ("transmitter", "receivers", "profile", "source_present"):
        if truth[key] != old_truth[key]:
            raise AssertionError(f"Unexpected change to {key}: {row['sample_id']}")
    for name, link in truth["links"].items():
        old = old_truth["links"][name]
        # These quantities must be identical; only antenna gain and derived
        # source powers/detections may change.
        for key, value in old.items():
            if key not in ("components", "event_censoring") and link[key] != value:
                raise AssertionError(f"Unexpected propagation/noise change: {name}/{key}")
        delta = link["rx_antenna_response"]["effective_gain_dbi"]
        if len(link["components"]) != len(old["components"]):
            raise AssertionError("Emission component count changed")
        for new_c, old_c in zip(link["components"], old["components"]):
            for key, value in old_c.items():
                if key != "median_rx_total_component_dbm" and new_c[key] != value:
                    raise AssertionError(f"Unexpected emission change: {name}/{key}")
            # Upstream component power traces use float32 before taking medians.
            observed = new_c["median_rx_total_component_dbm"] - old_c["median_rx_total_component_dbm"]
            if abs(observed - delta) > 5e-5:
                raise AssertionError(f"Antenna attenuation mismatch: {observed} vs {delta}")


def baseline_checks(freq, models, config, baseline, patterns):
    """Verify one original scene per split before writing the new dataset."""
    base.initialize_worker(freq, models, config)
    reports = []
    for split in SPLITS:
        tables = baseline[split]
        truth = tables["ground_truth.jsonl"][0]
        scenario = {k: truth[k] for k in SCENARIO_FIELDS}
        row = base.simulate_scene(scenario)
        if {"sample_id": row["sample_id"], **row["input"]} != tables["inputs.jsonl"][0]:
            raise AssertionError(f"Original omnidirectional incidents do not reproduce: {split}")
        if row["truth"] != truth:
            raise AssertionError(f"Original omnidirectional ground truth does not reproduce: {split}")
        zero = {name: {**p, "max_attenuation_db": 0.0} for name, p in patterns.items()}
        initialize_worker(freq, models, config, zero)
        replay = simulate_directional(scenario)
        if replay["input"] != row["input"] or replay["label"] != row["label"]:
            raise AssertionError("Zero-attenuation adapter changes observations/labels")
        verify_physics(replay, truth)
        reports.append({"split": split, "sample_id": scenario["sample_id"],
                        "original_records_match": True, "zero_attenuation_adapter_matches": True})
    return reports


def fingerprint_files(paths):
    return {str(p): base.sha256(p) for p in paths}


def write_report(out, config, summary):
    p = config["receiver_pattern"]
    lines = ["# Directional receiver dataset", "",
        "This dataset replays the 1,400 selected v3 scenes with fixed directional receiving antennas.",
        "The source positions, IDs, labels, receiver noise seeds and geographic splits are unchanged.",
        "The 4,000-scene proposal pool was not regenerated. No new detection-based selection was applied.", "",
        "## Antennas", "",
        f"Peak gain: {p['peak_gain_dbi']} dBi. Horizontal half-power beamwidth: {p['horizontal_hpbw_deg']} degrees. "
        f"Maximum attenuation: {p['max_attenuation_db']} dB. No vertical or frequency-dependent pattern.",
        "Both antennas point permanently toward the original sampling-region center, using WGS84 bearings.", ""]
    for name, pattern in config["resolved_patterns"].items():
        lines.append(f"- {name}: azimuth {pattern['azimuth_deg']:.6f} degrees clockwise from north.")
    lines += ["", "## Detection coverage", "",
        "| Split | Samples | Both, v3 -> directional | Any, v3 -> directional | Neither, directional |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for split, data in summary["splits"].items():
        a, b = data["baseline"], data["directional"]
        lines.append(f"| {split} | {b['samples']} | {a['both']} -> {b['both']} | "
                     f"{a['samples']-a['neither']} -> {b['samples']-b['neither']} | {b['neither']} |")
    lines += ["", "## Reading the files", "",
        "Each split has inputs.jsonl, labels.jsonl and ground_truth.jsonl, joined by sample_id. "
        "The input/label schemas are unchanged. Empty incident lists are retained, including scenes where both receivers detect nothing.",
        "A fully empty scene has no signal evidence for a useful point-location estimate; report it as no detection "
        "or evaluate detected scenes separately while also reporting overall coverage. Do not silently drop and replace it.", "",
        "In ground truth, receivers[].antenna_gain_dbi remains the fixed 0 dBi boresight gain. "
        "receiver_antenna_patterns stores the fixed settings. links.RX*.rx_antenna_response stores the source bearing, "
        "off-axis angle and effective gain for that scene. These source-dependent fields are audit-only, not model inputs.", "",
        "The direction-dependent gain is applied to the source before receiver noise and incident detection. "
        "Noise figure, noise realizations, propagation, transmitter emissions and detector settings are unchanged.",
        "This is a controlled horizontal attenuation model, not a measured commercial antenna pattern or an angle-of-arrival sensor.", "",
        "## Checks", "",
        "Three original scenes reproduce exactly (one per split); setting adapter attenuation to zero also reproduces their observations. "
        "All new scenes pass paired checks for unchanged labels, geometry, propagation, noise and source emissions, "
        "and for the expected antenna attenuation in component power. Labels match v3 byte-for-byte.",
        "No model was trained or selected using this run. Coverage statistics do not establish a localization improvement.", "",
        "See config.json, manifest.json, verification.json, summary.json and comparison.csv for parameters and checks.",
        "The archive contains this complete dataset and replay script snapshots, but not IQ, terrain or the upstream dependency.", ""]
    (out / "README.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/directional_receivers_v4.json")
    parser.add_argument("--output", type=Path, help="Override output, relative to shell cwd")
    args = parser.parse_args(argv)
    started = time.monotonic()
    request = json.loads(args.config.read_text())
    if set(request) != {"name", "baseline_dataset", "output", "srtm_dir", "workers", "receiver_pattern"}:
        raise ValueError("Unknown or missing top-level config fields")
    if not isinstance(request["workers"], int) or request["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    source = (ROOT / request["baseline_dataset"]).resolve()
    out = args.output.resolve() if args.output else (ROOT / request["output"]).resolve()
    archive = out.with_suffix(".zip")
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Output must be separate from the baseline dataset")
    if out.exists() or archive.exists():
        raise FileExistsError("Output or archive exists; select a new output directory")
    source_config = json.loads((source / "source_simulation_config.json").read_text())
    patterns = resolve_patterns(source_config, request["receiver_pattern"])
    config = {**source_config, "record_format": "v2", "srtm_dir": str((ROOT / request["srtm_dir"]).resolve()),
              "start_utc": source_config.get("start_utc", base.sim.SIM_START_UTC)}
    if base.sha256(base.SIM_SOURCE) != source_config["simulator_sha256"]:
        raise ValueError("Simulator source differs from the baseline")
    if base.sha256(ROOT / "EMILY-X/srsran_loaded.cf32") != source_config["iq_sha256"]:
        raise ValueError("Loaded IQ differs from the baseline")
    expected_rxs = [asdict(replace(rx, height_agl_m=config["profile"]["rx_height_m"]))
                    for rx in base.ORIGINAL_RXS]
    if expected_rxs != source_config["receivers"]:
        raise ValueError("Receiver hardware differs from the baseline")
    baseline = load_baseline(source, source_config)
    protected = [Path(base.__file__), base.SIM_SOURCE]
    protected += [p for p in sorted(source.rglob("*")) if p.is_file()]
    before = fingerprint_files(protected)
    manifest = base.runtime_manifest(config)
    reference = json.loads((ROOT / "configs/reference_v3_inputs.json").read_text())
    if manifest["terrain_sha256"] != reference["terrain_sha256"]:
        raise ValueError("Terrain differs from the v3 reference")
    manifest.update(protected_files_sha256=before, adapter_sha256=base.sha256(__file__),
                    config_sha256=base.sha256(args.config), iq_sha256=source_config["iq_sha256"])
    print("Building original loaded-IQ templates and checking baseline reproduction...", flush=True)
    freq, models = base.build_models()
    checks = baseline_checks(freq, models, config, baseline, patterns)
    print("Three baseline and zero-attenuation replays passed.", flush=True)
    out.mkdir(parents=True)
    resolved = {**request, "baseline_dataset": str(source), "output": str(out),
                "resolved_patterns": patterns, "simulation": config,
                "sampling": "Replay selected v3 scenes in their original split and order; retain empty scenes"}
    base.write_json(out / "config.json", resolved)
    base.write_json(out / "manifest.json", manifest)
    shutil.copyfile(__file__, out / "directional_generator_snapshot.py")
    shutil.copyfile(base.__file__, out / "base_generator_snapshot.py")
    shutil.copyfile(source / "split_manifest.json", out / "baseline_split_manifest.json")
    scenarios = [{k: truth[k] for k in SCENARIO_FIELDS} for split in SPLITS
                 for truth in baseline[split]["ground_truth.jsonl"]]
    old_truths = {r["sample_id"]: r for split in SPLITS for r in baseline[split]["ground_truth.jsonl"]}
    rows = {}
    try:
        with futures.ProcessPoolExecutor(max_workers=request["workers"],
                mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker,
                initargs=(freq, models, config, patterns)) as pool:
            jobs = {pool.submit(simulate_directional, s): s["sample_id"] for s in scenarios}
            for i, job in enumerate(futures.as_completed(jobs), 1):
                row = job.result()
                if row["sample_id"] != jobs[job]:
                    raise AssertionError("Worker returned an unexpected sample")
                verify_physics(row, old_truths[row["sample_id"]])
                rows[row["sample_id"]] = row
                if i % 100 == 0 or i == len(scenarios):
                    print(f"{i}/{len(scenarios)} directional scenes; {time.monotonic()-started:.1f} s", flush=True)
    except Exception as exc:
        base.write_json(out / "FAILED.json", {"error": repr(exc), "completed": len(rows)})
        raise
    summary = {"splits": {}}
    import csv
    with (out / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "split", "latitude_deg", "longitude_deg", "baseline_RX1_detected",
                         "baseline_RX2_detected", "directional_RX1_detected", "directional_RX2_detected",
                         "RX1_off_axis_deg", "RX2_off_axis_deg", "RX1_gain_dbi", "RX2_gain_dbi"])
        for split in SPLITS:
            tables = baseline[split]
            selected = [rows[r["sample_id"]] for r in tables["inputs.jsonl"]]
            base.export_rows(out / split, selected)
            if base.sha256(out / split / "labels.jsonl") != base.sha256(source / split / "labels.jsonl"):
                raise AssertionError("Label bytes changed")
            inputs = read_jsonl(out / split / "inputs.jsonl")
            summary["splits"][split] = {"baseline": coverage(tables["inputs.jsonl"]),
                                         "directional": coverage(inputs)}
            for old, row in zip(tables["inputs.jsonl"], selected):
                responses = [row["truth"]["links"][name]["rx_antenna_response"] for name in ("RX1", "RX2")]
                writer.writerow([row["sample_id"], split, row["label"]["latitude_deg"],
                    row["label"]["longitude_deg"], *map(int, flags(old)), *map(int, flags(row["input"])),
                    *[r["off_axis_deg"] for r in responses], *[r["effective_gain_dbi"] for r in responses]])
    if fingerprint_files(protected) != before:
        raise AssertionError("An original source or dataset file changed")
    verification = {"passed": True, "original_files_unchanged": True,
        "baseline_reproduction": checks, "paired_physics_checked": len(rows),
        "labels_byte_identical": True, "sample_order_and_split_preserved": True,
        "zero_detection_scenes_retained": True, "model_training_performed": False,
        "table_sha256": {f"{split}/{name}": base.sha256(out / split / name)
                         for split in SPLITS for name in TABLES}}
    base.write_json(out / "verification.json", verification)
    summary["wall_seconds_before_packaging"] = time.monotonic() - started
    base.write_json(out / "summary.json", summary)
    write_report(out, resolved, summary)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                z.write(path, arcname=str(Path(out.name) / path.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise AssertionError("Archive integrity check failed")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Dataset: {out}\nArchive: {archive}\nTotal wall time: {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
