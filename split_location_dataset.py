#!/usr/bin/env python3
"""Create fixed geographic train/validation/test sets from the v2 pool.

No RF simulation or model evaluation is performed. The source data are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
SIZES = {"train": 1000, "validation": 200, "test": 200}
WIDTH_M = 20000.
BLOCK_M = 4000.
BUFFER_M = 250.


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_table(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    table = {r["sample_id"]: r for r in rows}
    if len(table) != len(rows):
        raise ValueError(f"Duplicate IDs in {path}")
    return table


def block_and_margin(x, y):
    values = np.array([x, y], dtype=float) + WIDTH_M/2
    if not np.all((values >= 0) & (values < WIDTH_M)):
        raise ValueError("Source position outside the declared 20 km square")
    bx, by = (values // BLOCK_M).astype(int)
    offsets = values % BLOCK_M
    margin = float(np.min(np.concatenate([offsets, BLOCK_M-offsets])))
    return int(by*5+bx), margin


def assign_blocks(seed):
    # Define roles before looking at detections, intensities or model results.
    order = np.random.default_rng(seed).permutation(25).tolist()
    return {b: "test" if i < 4 else "validation" if i < 8 else "train"
            for i, b in enumerate(order)}


def select_samples(inputs, truth, previously_evaluated, seed):
    roles = assign_blocks(seed)
    eligible = {b: [] for b in range(25)}
    audit = {}
    for sid, row in truth.items():
        block, margin = block_and_margin(row["x_m"], row["y_m"])
        role = roles[block]
        has_incident = any(r["incidents"] for r in inputs[sid]["receivers"])
        reasons = []
        if not has_incident:
            reasons.append("no_incident")
        if margin < BUFFER_M:
            reasons.append("block_boundary_buffer")
        if role != "train" and sid in previously_evaluated:
            reasons.append("previously_used_for_model_comparison")
        if not reasons:
            eligible[block].append(sid)
        audit[sid] = {"sample_id": sid, "block_id": block, "assigned_region": role,
                      "selected_split": "", "boundary_margin_m": margin,
                      "previously_evaluated": sid in previously_evaluated,
                      "eligible": not reasons, "exclusion_reasons": ";".join(reasons)}
    selected = {}
    quotas = {}
    for role, count in SIZES.items():
        blocks = sorted(b for b in roles if roles[b] == role)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 1, list(SIZES).index(role)]))
        remainder_order = rng.permutation(blocks).tolist()
        quotas[role] = {b: count//len(blocks) + int(b in remainder_order[:count % len(blocks)]) for b in blocks}
        ids = []
        for block in blocks:
            candidates = sorted(eligible[block])
            take = quotas[role][block]
            if len(candidates) < take:
                raise ValueError(f"Block {block} has {len(candidates)} eligible samples for quota {take}; "
                                 "no seed search or silent replacement is performed")
            choose_rng = np.random.default_rng(np.random.SeedSequence([seed, 2, block]))
            choose_rng.shuffle(candidates)
            ids.extend(candidates[:take])
        rng.shuffle(ids)
        selected[role] = ids
        for sid in ids:
            audit[sid]["selected_split"] = role
    return selected, audit, roles, quotas, eligible


def verify_split(selected, roles, inputs, labels, truth, previous_ids):
    from scipy.spatial.distance import cdist
    ids_by_split = {k: set(v) for k, v in selected.items()}
    coords_by_split = {}
    for split, ids in selected.items():
        if len(ids) != SIZES[split] or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid size or duplicate IDs in {split}")
        if split != "train" and set(ids) & previous_ids:
            raise ValueError("Previously evaluated sample in a held-out split")
        positions = []
        for sid in ids:
            t, label, inp = truth[sid], labels[sid], inputs[sid]
            block, margin = block_and_margin(t["x_m"], t["y_m"])
            if roles[block] != split or margin < BUFFER_M:
                raise ValueError("Geographic block or buffer violation")
            if (label["latitude_deg"], label["longitude_deg"]) != (t["latitude_deg"], t["longitude_deg"]):
                raise ValueError("Label/truth coordinates differ")
            if len(inp["receivers"]) != 2 or not any(r["incidents"] for r in inp["receivers"]):
                raise ValueError("Missing receiver groups or no incidents")
            positions.append((t["x_m"], t["y_m"]))
        if len(set(positions)) != len(positions):
            raise ValueError("Repeated source positions")
        coords_by_split[split] = np.array(positions)
    distances = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids_by_split[a] & ids_by_split[b]:
            raise ValueError("IDs overlap across splits")
        minimum = float(cdist(coords_by_split[a], coords_by_split[b]).min())
        if minimum < 2*BUFFER_M-1e-6:
            raise ValueError("Source locations too close across splits")
        distances[f"{a}_to_{b}"] = minimum
    return distances


def export_tables(out, selected, source_paths):
    # Copy complete JSONL rows byte-for-byte, preserving all receiver incidents.
    hashes = {}
    for role, ids in selected.items():
        dest = out / role
        dest.mkdir()
        hashes[role] = {}
        for name, path in source_paths.items():
            lines = {json.loads(line)["sample_id"]: line for line in path.read_text().splitlines()}
            data = "\n".join(lines[sid] for sid in ids) + "\n"
            target = dest / name
            target.write_text(data)
            hashes[role][name] = sha256(target)
            if read_table(target) != {sid: json.loads(lines[sid]) for sid in ids}:
                raise ValueError("Export differs from source records")
    return hashes


def plot_split(out, selected, truth, roles):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    colors = {"train": "#2878ad", "validation": "#de9223", "test": "#aa4586"}
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for block, role in roles.items():
        bx, by = block % 5, block // 5
        corner = (bx*4-10, by*4-10)
        ax.add_patch(Rectangle(corner, 4, 4, facecolor=colors[role], alpha=.13, edgecolor="none"))
        ax.add_patch(Rectangle((corner[0]+.25, corner[1]+.25), 3.5, 3.5,
                               facecolor="none", edgecolor=colors[role], linewidth=.6))
        ax.text(corner[0]+.13, corner[1]+.12, f"B{block:02d}", fontsize=8, color="#333333")
    for role, ids in selected.items():
        xy = np.array([[truth[s]["x_m"], truth[s]["y_m"]] for s in ids])/1000
        ax.scatter(xy[:, 0], xy[:, 1], s=9, c=colors[role], alpha=.75)
    ax.set(xlim=(-10, 10), ylim=(-10, 10), aspect="equal",
           xlabel="East of v2 region center (km)", ylabel="North of v2 region center (km)",
           title="Fixed geographic split: 1,000 train / 200 validation / 200 test")
    ax.set_xticks(np.arange(-10, 11, 4))
    ax.set_yticks(np.arange(-10, 11, 4))
    ax.grid(alpha=.3)
    ax.legend(handles=[Patch(color=colors[r], label=f"{r}: {len(ids)}") for r, ids in selected.items()],
              loc="upper left", fontsize=9)
    fig.savefig(out / "split_map.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/terrestrial_location_v2/campaign")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/terrestrial_location_v3")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if out.exists():
        parser.error("Output exists: use a new directory. Existing splits are never overwritten")
    if out == source or source in out.parents:
        parser.error("Output must be separate from the read-only source campaign")
    paths = {n: source/n for n in ("inputs.jsonl", "labels.jsonl", "ground_truth.jsonl")}
    inputs, labels, truth = (read_table(p) for p in paths.values())
    if not inputs.keys() == labels.keys() == truth.keys():
        raise ValueError("Source tables have different IDs")
    source_config = json.loads((source / "config.json").read_text())
    if source_config["width_km"] != 20.:
        raise ValueError("This split design requires the 20 km v2 region")
    previous_path = source / "incident_training/selection.json"
    previous_ids = set(json.loads(previous_path.read_text())["sample_ids"])
    # Include any other IDs actually evaluated by the saved diagnostic, rather
    # than relying solely on the old training-folder name.
    diagnostic = source / "analysis/diagnostic_predictions.csv"
    if diagnostic.exists():
        with diagnostic.open() as f:
            previous_ids.update(r["sample_id"] for r in csv.DictReader(f))
    source_checksums = {str(p.relative_to(ROOT)): sha256(p) for p in
                        [*paths.values(), source/"config.json", previous_path, diagnostic] if p.exists()}
    selected, audit, roles, quotas, eligible = select_samples(inputs, truth, previous_ids, args.seed)
    distances = verify_split(selected, roles, inputs, labels, truth, previous_ids)
    out.mkdir(parents=True)
    hashes = export_tables(out, selected, paths)
    shutil.copyfile(__file__, out / "splitter_snapshot.py")
    shutil.copyfile(source/"config.json", out / "source_simulation_config.json")
    manifest = {"version": 3, "source": str(source), "split_seed": args.seed,
                "splitter_sha256": sha256(__file__), "source_sha256": source_checksums,
                "sizes": SIZES, "block_width_m": BLOCK_M, "boundary_buffer_each_side_m": BUFFER_M,
                "block_roles": roles, "quotas_per_block": quotas, "sample_ids": selected,
                "previously_evaluated_ids": sorted(previous_ids), "output_sha256": hashes,
                "policy": "Assign 25 blocks using a fixed seed; exclude boundary buffers; select equal counts per role's blocks; held-out IDs exclude prior model comparisons.",
                "split_unit": "Source location; all receiver incidents and any repeats at that location belong together",
                "test_model_evaluation_performed": False,
                "test_scope": "Held-out positions/blocks for new models trained from scratch; same simulator, waveform and hardware as training. Parent pool was previously inspected for coverage."}
    write_json(out/"split_manifest.json", manifest)
    with (out/"allocation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(audit.values()))))
        writer.writeheader()
        writer.writerows(audit.values())
    write_json(out/"unselected_sample_ids.json", [sid for sid, row in audit.items() if not row["selected_split"]])
    stats = {}
    for role, ids in selected.items():
        counts = {"RX1_only": 0, "RX2_only": 0, "both": 0}
        for sid in ids:
            a, b = [bool(g["incidents"]) for g in inputs[sid]["receivers"]]
            counts["both" if a and b else "RX1_only" if a else "RX2_only"] += 1
        stats[role] = {"samples": len(ids), "blocks": sorted(b for b in roles if roles[b] == role),
                       "eligible_before_subsampling": sum(len(eligible[b]) for b in roles if roles[b] == role),
                       "previous_comparison_overlap": len(set(ids) & previous_ids),
                       "detections": counts}
    for relative, expected in source_checksums.items():
        if sha256(ROOT/relative) != expected:
            raise ValueError("Source data changed during export")
    report = {"passed": True, "splits": stats, "source_samples": len(inputs),
              "source_detected_samples": sum(any(g["incidents"] for g in r["receivers"]) for r in inputs.values()),
              "unused_source_samples": len(audit)-sum(SIZES.values()),
              "minimum_cross_split_distance_m": distances, "pairwise_id_overlap": 0,
              "pairwise_block_overlap": 0, "held_out_prior_comparison_overlap": 0,
              "records_equal_source": True, "source_files_unchanged": True,
              "model_fit_or_test_scoring_performed": False}
    write_json(out/"verification.json", report)
    plot_split(out, selected, truth, roles)
    text = ["# LTE location dataset v3 — fixed geographic split", "",
            "Use this version for new training runs. It replaces the old 1,000-example selection as the training entry point. The parent v2 simulation is unchanged.", "",
            "| Set | Samples | Geographic blocks | Purpose |", "|---|---:|---:|---|",
            "| train | 1,000 | 17 | Fit model weights and preprocessing |",
            "| validation | 200 | 4 | Select models, settings and stopping point |",
            "| test | 200 | 4 | Score the final frozen model |", "",
            "## How it was split", "",
            "The v2 20 × 20 km region is divided into 25 blocks, each 4 × 4 km. A fixed seed assigns each entire block to one set before sampling. No block is shared.",
            "Points within 250 m of any block boundary are excluded, leaving at least 500 m between samples across sets. The test therefore covers the interiors of held-out blocks.",
            "We then sample roughly equal counts from each assigned block: 58 or 59 per training block, 50 per validation/test block. Only source windows containing incidents are included; dual-receiver detection is not required.",
            "Both validation and test exclude the old 1,000 positions used for model comparisons. Other positions in their blocks are also excluded from the new training set.", "",
            "## Use", "",
            "Each set has `inputs.jsonl`, `labels.jsonl` and `ground_truth.jsonl`. Join on `sample_id`. One sample keeps both receivers and all of their incidents together.",
            "Model input comes only from `inputs.jsonl`. Source coordinates are in `labels.jsonl`; `ground_truth.jsonl` is for audit only. IDs and block assignments are not model features.",
            "Fit normalization and feature selection on train only. Tune on validation; use test after those choices are fixed. Start a new model: weights trained on the old dataset may already have seen locations in the held-out regions.",
            "No model was fitted or scored as part of making this split. The parent pool was already checked for coverage, so this is a fixed holdout for future modeling, not a wholly untouched simulation campaign. It does not test new receivers, waveform families or real data.", "",
            "## Checks", "",
            "All three sets have disjoint IDs, positions and blocks. Validation/test contain no previously compared sample IDs. Exported records match the parent rows exactly.",
            f"Minimum source distances: train–validation {distances['train_to_validation']:.0f} m; train–test {distances['train_to_test']:.0f} m; validation–test {distances['validation_to_test']:.0f} m.", "",
            "- `split_manifest.json`: fixed block roles, sample IDs, seed, source/output hashes.",
            "- `allocation.csv`: the role and selection/exclusion reason for every one of the 4,000 source samples. Audit only.",
            "- `verification.json`: counts and isolation checks.",
            "- `split_map.png`: the three geographic partitions and the empty boundary strips.",
            "- Unselected records remain in the parent pool. Do not add records from validation/test blocks to training later.", "",
            "## Reproduce", "", "From the workspace root, using a new output folder:", "", "```bash",
            ".venv/bin/python split_location_dataset.py \\",
            "  --source datasets/terrestrial_location_v2/campaign \\",
            "  --output datasets/terrestrial_location_v3_copy --seed 20260922", "```", ""]
    (out/"README.md").write_text("\n".join(text))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
