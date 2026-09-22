#!/usr/bin/env python3
"""Unified terrestrial localization generator using the unmodified v3.2 simulator.

Run with .venv/bin/python generate_location_dataset.py --help.
Each JSONL row is one source position and one simultaneous two-receiver window.
No detection-based rejection, no replacement of failed simulations with empty data.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import argparse
import concurrent.futures as futures
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import multiprocessing
import shutil
import subprocess
import sys
import time

import numpy as np
from pyproj import CRS, Transformer

sys.path.insert(0, str(ROOT / "Incident_simulator"))
import emilyx_lte_end_to_end_simulator_v3_2 as sim

VERSION = 3  # Generator version; the dataset split is a separate operation.
BASE_TX = replace(sim.TX)
ORIGINAL_TX = replace(BASE_TX)
ORIGINAL_RXS = tuple(replace(rx) for rx in sim.RECEIVERS)
SIM_SOURCE = Path(sim.__file__)
PROFILES = {
    "original": {"region": "original", "tx_height_m": 40., "rx_height_m": 2., "downtilt_deg": 4.},
    "original_tilt0": {"region": "original", "tx_height_m": 40., "rx_height_m": 2., "downtilt_deg": 0.},
    "original_rx30": {"region": "original", "tx_height_m": 40., "rx_height_m": 30., "downtilt_deg": 4.},
    "original_tx80_rx30": {"region": "original", "tx_height_m": 80., "rx_height_m": 30., "downtilt_deg": 4.},
    "near_midpoint": {"region": "midpoint", "south_km": 0., "tx_height_m": 40., "rx_height_m": 2., "downtilt_deg": 4.},
    "near_south10": {"region": "midpoint", "south_km": 10., "tx_height_m": 40., "rx_height_m": 2., "downtilt_deg": 4.},
    "near_south20": {"region": "midpoint", "south_km": 20., "tx_height_m": 40., "rx_height_m": 2., "downtilt_deg": 4.},
}

# Explicit allowlist: receiver coordinates are observations, source coordinates
# and all simulator metadata are labels/truth. IDs are join keys only.
INCIDENT_FIELDS = (
    "data_type", "t_start", "t_mid", "t_end", "t_start_offset_sec",
    "t_mid_offset_sec", "t_end_offset_sec", "duration_sec", "ongoing",
    "time_structure", "f_low_hz", "f_high_hz", "f_center_hz", "bw_hz",
    "intensity_kind", "intensity_value", "lat", "lon", "altitude",
    "file_freq_low_hz", "file_freq_high_hz", "file_freq_resolution_hz",
    "file_time_resolution_sec", "observation_start", "observation_end",
    "observation_duration_sec", "n_time_samples", "n_freq_channels",
    "n_pixels", "n_time_pixels", "n_freq_pixels",
)
WORKER = {}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sim.sanitize_json_value(value), indent=2,
                              allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(sim.sanitize_json_value(row), allow_nan=False) + "\n")
    tmp.replace(path)


def projections():
    local = CRS.from_proj4(f"+proj=aeqd +lat_0={ORIGINAL_TX.lat_deg} "
                          f"+lon_0={ORIGINAL_TX.lon_deg} +datum=WGS84 +units=m")
    return (Transformer.from_crs("EPSG:4326", local, always_xy=True),
            Transformer.from_crs(local, "EPSG:4326", always_xy=True))


def sample_scenarios(stage, count, cells, width_km, seed):
    """Equal-size local-grid strata; preserve historical random draw order."""
    if stage not in ("pilot", "train") or count < 1 or cells < 1 or width_km <= 0:
        raise ValueError("Invalid stage, count, grid or region width")
    if stage == "pilot" and count != cells * cells:
        raise ValueError("Pilot needs exactly one point per grid cell")
    if count % (cells * cells):
        raise ValueError("Count must be a multiple of cells squared")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    _, inverse = projections()
    width = width_km * 1000
    step = width / cells
    points = []
    for iy in range(cells):
        for ix in range(cells):
            for _ in range(count // (cells * cells)):
                offset = np.array([0.5, 0.5]) if stage == "pilot" else rng.random(2)
                x, y = -width / 2 + (np.array([ix, iy]) + offset) * step
                lon, lat = inverse.transform(x, y)
                points.append({"x_m": float(x), "y_m": float(y),
                               "latitude_deg": lat, "longitude_deg": lon,
                               "cell_x": ix, "cell_y": iy})
    # Randomize file order: IDs do not encode position or spatial cell.
    rng.shuffle(points)
    for i, row in enumerate(points):
        row["sample_id"] = f"{stage}_{i:06d}"
        stage_key = 20 if stage == "pilot" else 30
        row["receiver_seeds"] = [int(s) for s in np.random.SeedSequence(
            [seed, stage_key, i]).generate_state(len(ORIGINAL_RXS))]
    return points


def profile_center(profile):
    if profile["region"] == "original":
        return ORIGINAL_TX.lon_deg, ORIGINAL_TX.lat_deg
    if profile["region"] == "custom":
        return profile["longitude_deg"], profile["latitude_deg"]
    a, b = ORIGINAL_RXS
    az, _, d = sim.GEOD.inv(a.lon_deg, a.lat_deg, b.lon_deg, b.lat_deg)
    lon, lat, _ = sim.GEOD.fwd(a.lon_deg, a.lat_deg, az, d / 2)
    lon, lat, _ = sim.GEOD.fwd(lon, lat, 180., profile.get("south_km", 0.) * 1000)
    return lon, lat


def projection(profile):
    lon, lat = profile_center(profile)
    crs = CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m")
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def sample_positions(stage, count, profile, seed, width_km=20., cells=10,
                     min_distance_km=1., record_format="v2"):
    points = sample_scenarios(stage, count, cells, width_km, seed)
    if record_format == "v1":
        if profile != PROFILES["original"] or min_distance_km != 0:
            raise ValueError("v1 record format requires original profile and minimum distance 0")
        return points
    _, inverse = projection(profile)
    fallback = np.random.default_rng(np.random.SeedSequence([seed, 741]))
    for point in points:
        replacements = 0
        while True:
            lon, lat = inverse.transform(point["x_m"], point["y_m"])
            distances = [sim.GEOD.inv(lon, lat, rx.lon_deg, rx.lat_deg)[2] for rx in ORIGINAL_RXS]
            if min(distances) >= min_distance_km * 1000:
                break
            point["x_m"] = (-width_km/2 + (point["cell_x"] + fallback.random()) * width_km/cells) * 1000
            point["y_m"] = (-width_km/2 + (point["cell_y"] + fallback.random()) * width_km/cells) * 1000
            replacements += 1
            if replacements > 10000:
                raise ValueError("Unable to sample this cell outside the configured receiver exclusion")
        point["latitude_deg"], point["longitude_deg"] = lat, lon
        point["geometry_replacements"] = replacements
        # Preserve existing IDs. This is an output schema, not code versioning.
        point["sample_id"] = "v2_" + point["sample_id"]
    return points


def build_models():
    """Same five loaded-IQ templates and RF/RBW calculations as upstream main."""
    sim.configure_radio_profile("terrestrial")
    freq = np.arange(sim.FREQ_MIN_HZ, sim.FREQ_MAX_HZ + 0.5 * sim.CHANNEL_SPACING_HZ,
                     sim.CHANNEL_SPACING_HZ, dtype=np.float64)
    freq = freq[freq <= sim.FREQ_MAX_HZ + 1e-6]
    raw = sim.load_srsran_iq_state(
        ROOT / "EMILY-X/srsran_loaded.cf32", sim.SRSRAN_IQ_SAMPLE_RATE_HZ,
        sim.SRSRAN_MAX_CAPTURE_SECONDS, sim.LTE_TEMPLATE_WINDOW_MS,
        sim.LTE_TEMPLATE_HOP_MS, sim.LTE_MAX_TEMPLATES_PER_STATE)
    templates = []
    for item in raw["templates"]:
        rel_db = 20 * np.log10(item["raw_rms"] / raw["raw_rms"])
        comps, diag, _, _ = sim.build_emission_components(item["iq"], sim.SRSRAN_IQ_SAMPLE_RATE_HZ)
        comps = sim.scaled_components_for_load(comps, rel_db)
        tx_dbm, fractions = sim.build_tx_rbw_spectrum(freq, comps)
        templates.append({
            "template_index": int(item["template_index"]), "start_s": item["start_s"],
            "end_s": item["end_s"], "raw_rms": item["raw_rms"],
            "relative_power_db": float(rel_db), "iq_excerpt": item["iq_excerpt"],
            "components": comps, "rf_diag": diag, "fractions": fractions,
            "tx_rbw_dbm": tx_dbm.astype(np.float32)})
    rep_idx = int(np.argmin([abs(t["raw_rms"] - raw["raw_rms"]) for t in templates]))
    rep = templates[rep_idx]
    mean_mw = np.mean(np.stack([sim.dbm_to_mw(t["tx_rbw_dbm"]) for t in templates]), axis=0)
    model = {"source": raw["source"], "raw_rms": raw["raw_rms"],
             "relative_power_db": 0.0, "capture_seconds_used": raw["capture_seconds_used"],
             "templates": templates, "representative_template_index": rep_idx,
             **{k: rep[k] for k in ("components", "rf_diag", "fractions", "iq_excerpt")},
             "tx_rbw_dbm": sim.mw_to_dbm(mean_mw, floor_dbm=-220).astype(np.float32)}
    return freq, {"loaded": model}


def initialize_worker(freq, models, config):
    sim.configure_radio_profile("terrestrial")
    profile = config.get("profile", PROFILES["original"])
    tx = replace(ORIGINAL_TX, height_agl_m=profile["tx_height_m"],
                 electrical_downtilt_deg=profile["downtilt_deg"])
    sim.RECEIVERS = [replace(rx, height_agl_m=profile["rx_height_m"]) for rx in ORIGINAL_RXS]
    duration = config["duration_s"]
    state_schedule = np.full(duration, "loaded", dtype="U8")
    # Same traffic/template sequence at every location, independent of noise.
    template_schedule = sim.make_template_schedule(
        state_schedule, models,
        np.random.default_rng(np.random.SeedSequence([config["seed"], 40])))
    WORKER.clear()
    WORKER.update(freq=freq, models=models, config=config, tx=tx,
                  active=sim.build_active_union(freq, models),
                  state_schedule=state_schedule, template_schedule=template_schedule,
                  axes=sim.make_time_axes(duration, config.get("start_utc", sim.SIM_START_UTC)))


def simulate_scene(scenario, noise_only=False):
    """No detector-based filtering; a failed calculation raises an exception."""
    c, freq, models = WORKER["config"], WORKER["freq"], WORKER["models"]
    sim.TX = replace(WORKER["tx"], lat_deg=scenario["latitude_deg"],
                     lon_deg=scenario["longitude_deg"])
    t_rel, utc, local = WORKER["axes"]
    receivers, links = [], {}
    for i, rx in enumerate(sim.RECEIVERS):
        prop = sim.prepare_propagation(rx, freq, WORKER["active"], "p452",
                                       Path(c["srtm_dir"]), False, sim.P452_TIME_PERCENT)
        # The off state produces the same noise/background but adds no source.
        schedule = np.full(c["duration_s"], "off", dtype="U8") if noise_only else WORKER["state_schedule"]
        spec, truth = sim.make_receiver_spectrogram_v3(
            rx, sim.TX, freq, models, schedule, WORKER["template_schedule"], prop,
            c["duration_s"], np.random.default_rng(scenario["receiver_seeds"][i]))
        events, _ = sim.run_local_emily_windows(spec, freq, utc, local, t_rel)
        records = sim.events_to_candidate_incidents(
            events, rx, freq, utc, local, t_rel, "not_saved.npz",
            models["loaded"]["components"], truth, models["loaded"]["source"])
        segments = [{"start_s": 0., "end_s": float(c["duration_s"]),
                     "state": "off" if noise_only else "loaded"}]
        records = sim.patch_candidate_incidents_v32(
            records, truth, models, segments, "terrestrial", None, t_rel)
        safe = [{k: rec[k] for k in INCIDENT_FIELDS} for rec in records]
        receivers.append({"receiver_id": rx.name, "incidents": safe})
        link = {k: v for k, v in truth.items() if k not in ("shadowing_db_t", "component_truth")}
        link["components"] = [
            {k: v for k, v in comp.items() if not k.endswith("_t")}
            for comp in truth["component_truth"]]
        link["shadowing_db_t"] = truth["shadowing_db_t"]
        link["event_censoring"] = ([] if events.empty else events[
            ["event_id", "left_censored", "right_censored"]].to_dict("records"))
        links[rx.name] = link
    row = {
        "sample_id": scenario["sample_id"],
        "input": {"receivers": receivers},
        "label": {"transmitter_type": "terrestrial_lte", "latitude_deg": sim.TX.lat_deg,
                  "longitude_deg": sim.TX.lon_deg},
        "truth": {**scenario, "transmitter": asdict(sim.TX), "links": links,
                  "source_present": not noise_only},
    }
    row = sim.sanitize_json_value(row)
    if c.get("record_format", "v2" if "profile" in c else "v1") == "v2":
        row["truth"]["receivers"] = [asdict(rx) for rx in sim.RECEIVERS]
        row["truth"]["profile"] = c["profile"]
    if noise_only:
        row["label"] = None
    return row


def validate_row(row):
    """Validate the observation/label boundary and upstream censoring semantics."""
    assert set(row["input"]) == {"receivers"}
    assert len(row["input"]["receivers"]) == len(sim.RECEIVERS)
    for group, rx in zip(row["input"]["receivers"], sim.RECEIVERS):
        assert set(group) == {"receiver_id", "incidents"}
        assert group["receiver_id"] == rx.name
        for inc in group["incidents"]:
            assert set(inc) == set(INCIDENT_FIELDS)
            assert inc["lat"] == rx.lat_deg and inc["lon"] == rx.lon_deg
            assert inc["intensity_kind"] == "mW" and inc["intensity_value"] > 0
            assert inc["f_low_hz"] <= inc["f_center_hz"] <= inc["f_high_hz"]
            if inc["ongoing"]:
                assert all(inc[k] is None for k in ("t_end", "t_end_offset_sec", "duration_sec"))
    assert row["label"]["latitude_deg"] == row["truth"]["latitude_deg"]
    assert row["label"]["longitude_deg"] == row["truth"]["longitude_deg"]
    json.dumps(row, allow_nan=False)


def export_rows(out, rows):
    """Write separate observation, label and audit-only tables."""
    write_jsonl(out / "inputs.jsonl", ({"sample_id": r["sample_id"], **r["input"]} for r in rows))
    write_jsonl(out / "labels.jsonl", ({"sample_id": r["sample_id"], **r["label"]} for r in rows))
    write_jsonl(out / "ground_truth.jsonl", ({"sample_id": r["sample_id"], **r["truth"]} for r in rows))


def export_results(out, scenarios, incident_target=0):
    rows = [json.loads((out / "results" / (s["sample_id"] + ".json")).read_text())
            for s in scenarios]
    for row in rows:
        validate_row(row)
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert len({(r["label"]["latitude_deg"], r["label"]["longitude_deg"]) for r in rows}) == len(rows)
    export_rows(out, rows)
    detected = [r for r in rows if any(g["incidents"] for g in r["input"]["receivers"])]
    write_json(out / "incident_sample_ids.json", [r["sample_id"] for r in detected])
    counts = {"neither": 0, "RX1_only": 0, "RX2_only": 0, "both": 0}
    for r in rows:
        flags = [bool(g["incidents"]) for g in r["input"]["receivers"]]
        counts[("neither", "RX2_only", "RX1_only", "both")[2 * flags[0] + flags[1]]] += 1
    summary = {"scenarios": len(rows), "with_incidents": len(detected),
               "without_incidents": len(rows) - len(detected), "detection_counts": counts,
               "incidents": sum(len(g["incidents"]) for r in rows for g in r["input"]["receivers"]),
               "all_rows_valid": True, "selection": "all proposed locations retained"}
    write_json(out / "summary.json", summary)
    if incident_target:
        if len(detected) < incident_target:
            raise RuntimeError(f"Only {len(detected)} detected scenes for target {incident_target}. "
                               "All proposals saved. Increase count in a NEW campaign; no silent resampling.")
        # Input order was shuffled before simulation. Selection is independent of
        # completion order, location labels, intensity and model performance.
        selected = detected[:incident_target]
        export_rows(out / "incident_training", selected)
        write_json(out / "incident_training/selection.json", {
            "count": len(selected), "parent_proposals": len(rows),
            "parent_detected": len(detected), "parent_detection_fraction": len(detected)/len(rows),
            "policy": "First N scenes in pre-shuffled proposal order with at least one incident at either receiver.",
            "bias": "Detection-conditioned locations, not uniform over the original square. Empty windows remain in parent tables.",
            "all_training": True, "sample_ids": [r["sample_id"] for r in selected]})
    return summary


def count_detections(rows):
    counts = {"neither": 0, "RX1_only": 0, "RX2_only": 0, "both": 0}
    for row in rows:
        a, b = [bool(g["incidents"]) for g in row["input"]["receivers"]]
        counts[("neither", "RX2_only", "RX1_only", "both")[2*a+b]] += 1
    return counts


def config_for(args, profile_name=None, stage=None):
    name = profile_name or args.profile
    stage = stage or args.stage
    profile = dict(PROFILES[name]) if name != "custom" else {
        **PROFILES["original"], "region": "custom",
        "latitude_deg": args.center_lat, "longitude_deg": args.center_lon}
    for option in ("tx_height_m", "rx_height_m", "downtilt_deg"):
        value = getattr(args, option)
        if value is not None:
            profile[option] = value
    lon, lat = profile_center(profile)
    count = args.cells**2 if stage == "pilot" else args.count
    return {"generator_version": VERSION, "generator_sha256": sha256(__file__),
            "simulator_sha256": sha256(SIM_SOURCE), "stage": stage,
            "count": count, "cells": args.cells, "width_km": args.width_km,
            "duration_s": args.duration, "start_utc": args.start_utc, "seed": args.seed,
            "profile_name": name, "profile": profile, "record_format": args.record_format,
            "center_latitude_deg": lat, "center_longitude_deg": lon,
            "min_distance_to_receiver_m": args.min_distance_km * 1000,
            "reference_transmitter": asdict(ORIGINAL_TX),
            "receivers": [asdict(replace(rx, height_agl_m=profile["rx_height_m"])) for rx in ORIGINAL_RXS],
            "srtm_dir": str(args.srtm_dir.resolve()),
            "iq_sha256": sha256(ROOT / "EMILY-X/srsran_loaded.cf32"),
            "propagation": "unmodified upstream ITU-R P.452-16 + SRTM",
            "sampling": "cell centers (resample geometrically excluded points within cell)" if stage == "pilot" else "equal count per spatial cell; uniform within cell outside receiver exclusion",
            "loaded_state_only": True, "rejection_sampling": False,
            "incident_training_target": args.incident_target if stage == "train" else 0}


def runtime_manifest(config):
    return {"python": sys.version, "packages": {n: importlib.metadata.version(n) for n in
                ("numpy", "scipy", "pandas", "pyproj", "pycraf", "astropy")},
            "simulator_git_commit": subprocess.check_output(
                ["git", "-C", str(SIM_SOURCE.parent), "rev-parse", "HEAD"], text=True).strip(),
            "terrain_sha256": {str(f.relative_to(config["srtm_dir"])): sha256(f)
                for f in sorted(Path(config["srtm_dir"]).rglob("*.hgt"))},
            "incident_fields": list(INCIDENT_FIELDS),
            "sample_id": "Join key only. Do not use as a model feature.",
            "empty_receivers": "No incidents during a known shared observation window, not receiver downtime.",
            "fixed_time": config["start_utc"],
            "random_streams": {"position_offsets_and_shuffle": "SeedSequence([seed, 10])",
                "receiver_noise": "SeedSequence([seed, 20 if pilot else 30, proposal_index])",
                "template_schedule": "SeedSequence([seed, 40])",
                "geometric_resampling": "SeedSequence([seed, 741])"},
            "spectra": "Full upstream frequency grid; no RF/propagation shortcut. Arrays not saved.",
            "scope": "One source, loaded traffic; power, carrier and receiver sites fixed."}


def prepare_output(out, config, scenarios, resume=False):
    manifest = runtime_manifest(config)
    if out.exists():
        if not resume:
            raise ValueError("Output exists. Use a new directory or --resume")
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Resume configuration, source code or IQ differs from the recorded run")
        if json.loads((out / "manifest.json").read_text()) != manifest:
            raise ValueError("Resume software environment or terrain differs from the recorded run")
        if [json.loads(s) for s in (out / "scenarios.jsonl").read_text().splitlines()] != scenarios:
            raise ValueError("Resume scenario list differs from the recorded run")
    else:
        out.mkdir(parents=True)
        shutil.copyfile(__file__, out / "generator_snapshot.py")
        write_json(out / "config.json", config)
        write_jsonl(out / "scenarios.jsonl", scenarios)
        write_json(out / "manifest.json", manifest)


def run_batch(out, scenarios, freq, models, config, workers):
    (out / "results").mkdir(exist_ok=True)
    pending = [s for s in scenarios if not (out / "results" / (s["sample_id"] + ".json")).exists()]
    start = time.monotonic()
    if pending:
        # spawn keeps simulator's mutable TX and pycraf state isolated per worker.
        with futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker, initargs=(freq, models, config)) as pool:
            jobs = {pool.submit(simulate_scene, s): s for s in pending}
            for i, job in enumerate(futures.as_completed(jobs), 1):
                scenario = jobs[job]
                try:
                    row = job.result()
                    validate_row(row)
                    write_json(out / "results" / (scenario["sample_id"] + ".json"), row)
                except Exception as exc:
                    write_json(out / "failure.json", {"sample_id": scenario["sample_id"], "error": repr(exc)})
                    for future in jobs:
                        future.cancel()
                    raise
                if i % 100 == 0 or i == len(pending):
                    print(f"{config['profile_name']}: {i}/{len(pending)} new scenes; {time.monotonic()-start:.1f} s", flush=True)
    summary = export_results(out, scenarios, config["incident_training_target"])
    if config["record_format"] == "v2":
        summary.update(profile=config["profile"],
                       any_detection_fraction=summary["with_incidents"]/len(scenarios),
                       both_detection_fraction=summary["detection_counts"]["both"]/len(scenarios))
        write_json(out / "summary.json", summary)
    write_json(out / "completion.json", {"workers_this_invocation": workers,
               "new_scenes_this_invocation": len(pending), "elapsed_s": time.monotonic()-start,
               "output_sha256": {n: sha256(out / n) for n in
                   ("inputs.jsonl", "labels.jsonl", "ground_truth.jsonl", "scenarios.jsonl")}})
    (out / "failure.json").unlink(missing_ok=True)
    return summary


def parse_args(argv=None, default_profile="original", default_count=1000,
               default_incident_target=0, default_workers=4):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["pilot", "survey", "train"], required=True)
    p.add_argument("--profile", choices=[*PROFILES, "custom"], default=default_profile)
    p.add_argument("--center-lat", type=float)
    p.add_argument("--center-lon", type=float)
    p.add_argument("--tx-height-m", type=float)
    p.add_argument("--rx-height-m", type=float)
    p.add_argument("--downtilt-deg", type=float)
    p.add_argument("--record-format", choices=["v1", "v2"], default=None,
                   help="v2 preserves the schema/IDs used by the current v3 split")
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--incident-target", type=int, default=default_incident_target,
                   help="Also export first N detected scenes for historical selection/exclusion; 0 disables")
    p.add_argument("--cells", type=int, default=10)
    p.add_argument("--width-km", type=float, default=20.)
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--start-utc", default=sim.SIM_START_UTC)
    p.add_argument("--min-distance-km", type=float, default=None)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--workers", type=int, default=default_workers)
    p.add_argument("--srtm-dir", type=Path, default=ROOT / "srtm_data")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.record_format is None:
        args.record_format = "v1" if args.profile == "original" and args.stage != "survey" and all(
            getattr(args, k) is None for k in ("tx_height_m", "rx_height_m", "downtilt_deg")) else "v2"
    if args.min_distance_km is None:
        args.min_distance_km = 0. if args.record_format == "v1" else 1.
    if args.count is None:
        args.count = args.cells**2 if args.stage in ("pilot", "survey") else default_count
    numeric = [args.width_km, args.min_distance_km] + [v for v in
        (args.center_lat, args.center_lon, args.tx_height_m, args.rx_height_m, args.downtilt_deg) if v is not None]
    if not all(np.isfinite(v) for v in numeric):
        p.error("Numeric parameters must be finite")
    if args.count < 1 or args.cells < 1 or args.width_km <= 0 or args.duration < 2 or args.workers < 1 or args.seed < 0 or args.min_distance_km < 0:
        p.error("Invalid count, grid, width, duration, workers, seed or minimum distance")
    if args.count % args.cells**2 or (args.stage != "train" and args.count != args.cells**2):
        p.error("Train count must be a multiple of cells squared; pilot/survey count must equal cells squared")
    if args.incident_target < 0 or (args.stage == "train" and args.incident_target > args.count):
        p.error("Invalid incident target")
    if args.profile == "custom":
        if args.center_lat is None or args.center_lon is None or not -90 < args.center_lat < 90 or not -180 <= args.center_lon <= 180:
            p.error("custom requires --center-lat (-90,90) and --center-lon [-180,180]")
    elif args.center_lat is not None or args.center_lon is not None:
        p.error("Use --profile custom when supplying a center")
    if any(v is not None and v <= 0 for v in (args.tx_height_m, args.rx_height_m)):
        p.error("Antenna heights must be positive")
    if args.record_format == "v1" and (args.profile != "original" or args.min_distance_km != 0 or
            args.stage == "survey" or any(getattr(args, k) is not None for k in ("tx_height_m", "rx_height_m", "downtilt_deg"))):
        p.error("v1 format requires the unchanged original profile, minimum distance 0, and pilot/train")
    if args.stage == "survey" and (args.profile != default_profile or any(getattr(args, k) is not None for k in
            ("center_lat", "center_lon", "tx_height_m", "rx_height_m", "downtilt_deg"))):
        p.error("survey compares all seven presets; use pilot/train for profile overrides")
    if args.output.exists() and not args.resume:
        p.error("Output exists. Use a new directory or --resume")
    sim.make_time_axes(2, args.start_utc)  # Validate before creating a run.
    return args


def main(argv=None, **defaults):
    args = parse_args(argv, **defaults)
    out = args.output.resolve()
    names = list(PROFILES) if args.stage == "survey" else [args.profile]
    stage = "pilot" if args.stage == "survey" else args.stage
    plans = []
    # Validate all resumable configurations before spending time on RF templates.
    for name in names:
        config = config_for(args, name, stage)
        scenarios = sample_positions(stage, config["count"], config["profile"], args.seed,
                                     args.width_km, args.cells, args.min_distance_km, args.record_format)
        dest = out/name if args.stage == "survey" else out
        prepare_output(dest, config, scenarios, args.resume)
        plans.append((name, dest, config, scenarios))
    needs_models = any(not (dest/"results"/(s["sample_id"]+".json")).exists()
                       for _, dest, _, scenarios in plans for s in scenarios)
    freq, models = build_models() if needs_models else (None, None)
    summaries = {}
    for name, dest, config, scenarios in plans:
        summaries[name] = run_batch(dest, scenarios, freq, models, config, args.workers)
        print(json.dumps(summaries[name], indent=2), flush=True)
    if args.stage == "survey":
        write_json(out/"comparison.json", summaries)


if __name__ == "__main__":
    main()
