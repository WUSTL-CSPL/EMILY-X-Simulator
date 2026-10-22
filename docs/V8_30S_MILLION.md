# v8: two million-sample datasets at 30 seconds

Historical workstation run. For a new machine, use [HANDOFF.md](HANDOFF.md). The commands below require the original workstation's reference metadata and parent virtual environment.

This run creates two new datasets. Each has 1,000,000 accepted locations in the existing 20 × 20 km region: one with three omni receivers, the other with four. The observation window is 30 seconds. Existing scripts and datasets are kept intact.

## Settings

- 700,000 training, 150,000 validation, 150,000 test samples per dataset.
- Same geographic blocks and split roles as v8, with a 250 m buffer inside each block boundary and a 1 km exclusion around receiver sites.
- Same receiver locations, 0 dBi omni antennas and 2 m antenna heights.
- Same loaded LTE IQ, 800 MHz carrier, 10 MHz bandwidth, 40 m transmitter height, antenna pattern, terrain and ITU-R P.452 propagation.
- Each proposed location gets one independently drawn TX power, uniform in dBm from 33 to 43. That power is shared by its receivers and is constant over the scene.
- Each receiver gets an independent N(0, 2²) dB measurement offset, constant within the scene. Original thermal noise and time-varying shadowing remain enabled.
- Same v8 random seeds and proposal order. All signals and detections are recomputed for 30 seconds; the old 60-second scene cache is never reused.
- Every retained sample must have incidents at every included receiver. The three-receiver dataset does not require RX4 to detect. Selection therefore changes the distribution of positions and powers from the original proposal distribution.

The two datasets share many scenarios; they are comparisons, not independent data to concatenate. IDs may also overlap older v8 datasets. Always keep versions separate. Geographic split roles remain the same across versions.

## Files and commands

Run from `emily-location-dataset`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ../.venv/bin/python -u generate_multireceiver_v8_30s_million.py \
  --config configs/multireceiver_v8_30s_million.json
```

To resume an interrupted run, use the same command with `--resume`. Do not run two copies at once; a campaign lock prevents concurrent writers. SIGTERM or Ctrl+C requests a clean stop after the current batch or shard. An abrupt interruption loses at most the current uncommitted batch/shard.

New output directories:

- `datasets/terrestrial_location_v8_30s_million_3rx_20km_1000000`
- `datasets/terrestrial_location_v8_30s_million_4rx_20km_1000000`

Each split contains 10,000-sample parts, each with aligned `inputs.jsonl.gz`, `labels.jsonl.gz`, and `ground_truth.jsonl.gz`. Use inputs for model features and labels for targets. Ground truth is for auditing, not input features. Read parts in numeric order and join by `sample_id`. These are gzip streams; a second giant ZIP is unnecessary.

```python
import gzip
import json
from pathlib import Path

root = Path('datasets/terrestrial_location_v8_30s_million_3rx_20km_1000000')
for part in sorted((root / 'train').glob('part-*')):
    with gzip.open(part / 'inputs.jsonl.gz', 'rt') as inputs, \
         gzip.open(part / 'labels.jsonl.gz', 'rt') as labels:
        for observation, target in zip(inputs, labels, strict=True):
            x, y = json.loads(observation), json.loads(target)
            assert x['sample_id'] == y['sample_id']
            # Build features from x['receivers']; targets are coordinates in y.
```

At 1 Hz, 30 samples span timestamps 0 through 29. The upstream `observation_duration_sec` therefore remains 29; `n_time_samples` is 30. Ongoing incidents retain null end times and durations.

## Progress and completion

- Log: `runs/multireceiver_v8_30s_million.log`
- Current batch: `runs/multireceiver_v8_30s_million/progress.json`
- Run state: `runs/multireceiver_v8_30s_million/status.json`
- Proposal/cache checkpoint: `runs/multireceiver_v8_30s_million/scene_cache.sqlite`
- Source, configuration and dependency fingerprints: `runs/multireceiver_v8_30s_million/manifest.json`

Simulation runs with 16 processes, in batches of 512 candidates. Shared four-receiver simulations feed separately filtered three- and four-receiver selections. This saves repeated simulation without changing the first three receivers' observations. The runner first fills all split quotas, then exports and validates both datasets. The output directories may not exist yet during simulation.

Only a dataset's `COMPLETE.json` confirms all of its shards passed validation. The campaign's `COMPLETE.json` confirms both datasets are complete and the protected originals are unchanged. A running process or nonempty directory is not a completed dataset.

The runner checks free disk space before each batch and shard and stops with a recoverable checkpoint below 60 GiB. Keep the machine awake and WSL running. A reboot needs a manual resume; it does not restart automatically.

## Preflight validation

The separate smoke configuration generates 1,000 samples per receiver setup, with the same physics, selection and output format. Validation includes exact replay against the previous native-30-second experiment, stopping/resuming without repeating committed proposals, aligned shard IDs, receiver detection requirements, geographic split roles, power/noise settings and source-file hashes. Its files have a separate `_smoke` prefix and are not the requested million-sample deliverables.

Read current counts and a rough remaining simulation time with:

```bash
python3 scripts/status_v8_30s_million.py
```

Preflight result: both 1,000-sample exports passed; all 1,152 cached scenes exactly matched the previous native-30-second experiment. The initial 64 committed proposals were preserved after restart. Nineteen retained three-receiver samples had no RX4 detection, confirming that the three-receiver filter is independent of RX4. Details: `runs/multireceiver_v8_30s_million_smoke/preflight_verification.json`.

## Local operations

The original workstation used a separately configured progress monitor and a
completion-checked Windows shutdown helper. Those machine-specific files are
not part of the Git handoff. For a new portable campaign, use the verification
command in [HANDOFF.md](HANDOFF.md); it checks the exported datasets without
shutting down the computer.
