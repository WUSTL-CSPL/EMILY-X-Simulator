# Dataset generation handoff

Start with `generate_dataset.py`. This is the portable entry point for the current v8 terrestrial experiment: 30-second observations, original omni receivers, and separately filtered three- and four-receiver datasets. Historical v3–v7 experiments are still available; they are not the default workflow.

## Set up a new machine

Use Linux or WSL2 with Python 3.12. Clone this repository with its submodule:

```bash
git clone --recurse-submodules YOUR_REPOSITORY_URL emily-location-dataset
cd emily-location-dataset
```

Provide `EMILY-X/srsran_loaded.cf32` and its `.cf32.json` sidecar, and the cached `.hgt` files under `srtm_data/`, preserving the `J11/` subdirectory. These assets and the Python environment are not stored in Git. See [DATA.md](DATA.md). The separate transfer ZIP includes these assets; copy its `EMILY-X/` and `srtm_data/` folders here if you have that bundle. Its entry-point filenames differ, so use this document for the Git checkout.

Then install the pinned dependencies and check the inputs:

```bash
EMILYX_PYTHON=python3.12 bash setup_simulator.sh
.venv/bin/python scripts/check_inputs.py --mode dataset --reference-v3
```

Dependency installation needs internet access. Keep long runs on the Linux filesystem when using WSL. Do not move an existing campaign between the ZIP bundle and this checkout: resume checks include code hashes and paths.

Greg's simulator is pinned as a public Git submodule at `a6124824f653467173c845c4ff609d4d35d6902f`. No write access to his repository is needed. Avoid replacing it with the latest main branch during a campaign.

The new entry point restores the small metadata files needed under ignored `datasets/` and `runs/` from `reference/v8/`. It resolves terrain relative to this checkout and records the new machine's Python build. Existing metadata is checked and never overwritten. You can run this preparation explicitly:

```bash
.venv/bin/python scripts/prepare_v8_inputs.py
```

Core package versions, IQ, terrain and source hashes remain checked against the reference. The historical source configuration contains a 60-second default; the active request overrides it to 30 seconds, and the resulting manifest records the actual value.

## Run and verify

Start with the small smoke campaign. It produces 20 accepted samples for each receiver setup:

```bash
.venv/bin/python generate_dataset.py --config configs/portable_smoke.json
.venv/bin/python scripts/verify_portable_output.py --config configs/portable_smoke.json
```

Then choose a larger request:

```bash
# 1,000 samples per receiver setup
.venv/bin/python generate_dataset.py --config configs/portable_generate_1000.json

# 1,000,000 samples per receiver setup
.venv/bin/python -u generate_dataset.py --config configs/portable_generate_million.json
```

After an interruption, rerun the same command with `--resume`. Keep the configuration, dependencies, code and location unchanged within a campaign. Only start one writer. Ctrl+C or SIGTERM checkpoints after the current batch or export shard; an abrupt stop may repeat the uncommitted batch.

Use the matching configuration to verify completed exports:

```bash
.venv/bin/python scripts/verify_portable_output.py --config configs/portable_generate_million.json
```

This rereads gzip tables, verifies file hashes, counts and aligned IDs, and compares exported records with stored hashes wherever their IDs match the reference cases. Different hardware may produce floating-point differences; investigate reference mismatches rather than assuming bitwise reproducibility across every platform. A different sampling seed may have no matching reference IDs; check the reported `reference_replays` count.

## What stays fixed

- Original 20 × 20 km region and geographic block roles, with a 250 m inner block buffer and 1 km receiver exclusion.
- Loaded terrestrial LTE IQ: 800 MHz carrier and 10 MHz bandwidth; transmitter height 40 m.
- Original omni receiver model: 0 dBi gain and 2 m height. RX3/RX4 positions are specified in the request.
- SRTM terrain and ITU-R P.452 propagation, with no automatic terrain download.
- Scene TX power sampled uniformly in dBm from 33 to 43; one shared power across receivers, fixed during the scene.
- Independent per-receiver Gaussian measurement offsets with sigma 2 dB, fixed within the scene. Original thermal noise and time-varying shadowing remain enabled.
- Every included receiver must detect an incident. The three-receiver selection does not require RX4 detection.

The million-sample request has 700,000 training, 150,000 validation and 150,000 test samples per receiver setup. These are accepted samples after detection filtering, not unconditionally uniform positions. The two receiver variants overlap and should not be concatenated as independent data.

For a new experiment, copy a request and use new `campaign_dir` and `output_prefix` values. Set `small_size` and `large_size` to the same target for this two-output workflow, and keep both split dictionaries equal with totals matching the target. Worker count affects resources, not the intended scene distribution. A new `sampling_seed` generates different candidate positions and randomization. Other durations, geographic regions, satellite sources and directional receivers need separate implementation and validation.

For two million samples per receiver setup, use split sizes 1,400,000 / 300,000 / 300,000; for five million, use 3,500,000 / 750,000 / 750,000. These work as separate fresh campaigns. Reusing one campaign to export two million and then extend to five million needs a new runner: the current entry point requires equal small/large targets and locks its configuration for resume. With the same seed and settings, the smaller per-split selections are prefixes of the larger ones; they are not independent datasets.

## Files and outputs

The call chain is:

```text
generate_dataset.py
  -> generate_multireceiver_v8_30s_million_profiled.py (timed batches)
  -> generate_multireceiver_v8_30s_million.py (checkpoint/export helpers)
  -> generate_multireceiver_dataset_v8.py (power/noise/scenario logic)
  -> generate_location_dataset.py
  -> Incident_simulator/emilyx_lte_end_to_end_simulator_v3_2.py
```

Each output split contains parts with `inputs.jsonl.gz`, `labels.jsonl.gz` and `ground_truth.jsonl.gz`. Join them by `sample_id`. Train with inputs and labels; ground truth is for auditing. See `docs/DATA_FIELDS_en.md` for fields. Ongoing incident end times remain null. Thirty 1 Hz samples span timestamps 0–29, so `observation_duration_sec=29` is expected.

The generator fills all split quotas before exporting the two variants. Completion requires campaign and dataset `COMPLETE.json` files, not just enough cached scenes. `batch_timings.jsonl` records simulation collection, encoding, validation, transaction costs, worker CPU/wall time and memory. Some measurements overlap; do not sum all fields as separate phases.

The previous machine's completed million-sample pair used about 9.93 GiB including the shared checkpoint, of which 4.65 GiB was compressed outputs. This is a measured end state, not a peak guarantee. The 60 GiB free-space threshold is a protective reserve.

## Previous slowdown and current limits

A previous long run slowed from roughly 16 seconds to 450 seconds per 512 candidates, then was interrupted by an unclean Windows reboot. A post-reboot diagnostic measured 15.7 seconds for a complete 512-scene simulation and about 0.016 seconds for a matching database transaction on a disposable full-size copy. The exact slowdown cause was not established. Timings were added rather than declaring a speculative fix. Do not reuse the old mixed historical ETA when estimating a resumed run.

Earlier directional and multi-receiver experiments are described in the versioned English documents listed in [PUSH_CONTENTS.md](PUSH_CONTENTS.md). Some older scripts and integration tests require old datasets or caches supplied separately. `scripts/status_v8_30s_million.py` is tied to the old campaign path; for portable runs, read the files in the request's `campaign_dir`. Estimate remaining time from recent batches in `batch_timings.jsonl`, not the historical helper's mixed average.

No shutdown automation is part of this handoff. The Windows shutdown helpers and machine-specific scheduling from the original workspace remain local.
