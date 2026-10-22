# Files for the next commit

The current Git history contains the v3 workflow. This submission adds the
later experiments and the portable v8 entry point. Files stay in their existing
locations so imports and source-hash checks still work.

## Start here

| Document | Purpose |
| --- | --- |
| [HANDOFF.md](HANDOFF.md) | New-machine setup, 30-second generation, resume, verification and larger requests |
| [DATA.md](DATA.md) | Required external IQ and terrain files |
| [DATA_FIELDS_en.md](DATA_FIELDS_en.md) | Incident fields and how to use inputs, labels and ground truth |
| [MULTIRECEIVER_DATASETS_V8.md](MULTIRECEIVER_DATASETS_V8.md) | Random power, Gaussian receiver offsets, receiver sites and filtering |
| [V8_30S_MILLION.md](V8_30S_MILLION.md) | Original million-sample experiment and checks |
| [V8_30S_PROFILED_RESUME.md](V8_30S_PROFILED_RESUME.md) | Historical resume procedure and performance measurements |

The README links to this handoff. All documents in this submission are in English.

## Current generation code

- `generate_dataset.py`: portable entry point for the two 30-second, 20 km datasets.
- `generate_multireceiver_v8_30s_million.py`: checkpoint, selection and gzip export helpers.
- `generate_multireceiver_v8_30s_million_profiled.py`: timed simulation batches.
- `generate_multireceiver_dataset_v8.py` and `multireceiver_campaign_v8.py`: v8 random power, receiver offsets and campaign logic.
- `scripts/prepare_v8_inputs.py`: restores small reference metadata in a fresh checkout.
- `scripts/verify_portable_output.py`: checks completed portable exports.
- `configs/portable_smoke.json`, `portable_generate_1000.json` and `portable_generate_million.json`: new-machine requests.
- `reference/v8/` and `PROVENANCE.json`: source configuration, geographic split roles, package/asset hashes, record references and upstream commit.

`generate_location_dataset.py`, `requirements.lock.txt`, `setup_simulator.sh`
and the pinned `Incident_simulator` submodule were already tracked. They remain
dependencies; the simulator source is not copied into the parent repository.

## Earlier experiments included for reference

| Version or task | New files included |
| --- | --- |
| v4, directional attenuation | `generate_directional_location_dataset.py`, its config, test and `DIRECTIONAL_DATASET.md` |
| v5, antenna gain and pattern | `generate_real_antenna_dataset_v5.py`, `receiver_antenna_v5.py`, antenna configs, digitizer, test and `REAL_ANTENNA_DATASET_V5.md` |
| v6, region comparison | `generate_location_comparison_v6.py`, analysis, config, test and `DATASET_COMPARISON_V6.md` |
| v7, three/four receivers | `generate_multireceiver_dataset_v7.py`, campaign, analysis, config, test and `MULTIRECEIVER_DATASETS_V7.md` |
| v8, power/noise and scale | v8 analysis, `generate_multireceiver_v8_large.py`, original requests and tests |
| 20/30/60-second comparison | `compare_v8_duration.py`, `analyze_v8_duration.py`, `configs/duration_ml_requirements.txt` |
| Original 30-second preflight | Original million/profiled smoke requests, status helper and preflight verifier |

These older analysis and preflight scripts may need datasets or caches from the
original workstation. The duration analysis also needs the packages listed in
`configs/duration_ml_requirements.txt`. Use the portable entry point for a fresh
v8 generation. Do not run every historical script as part of setup.

## Files that stay local

- IQ, terrain, generated datasets, SQLite caches, environments, logs and ZIP archives are ignored.
- `scripts/shutdown_after_v8_30s_*.py` and their two tests stay local.
- `generate_multireceiver_v8_30s_million_uncompressed.py`, its smoke request and `docs/V8_30S_UNCOMPRESSED.md` are superseded by gzip export.
- `docs/DATA_FIELDS_zh.md` stays local; the English field guide is tracked.

The Git handoff and the standalone transfer ZIP are separate deliverables. Git
contains code and small references; the ZIP also supplies the external IQ and
terrain. The ZIP itself is not part of this submission.

## Handoff checks

On October 22, 2026, the staged files were copied to a separate folder with no
Git metadata, together with the pinned simulator and links to the external
IQ/terrain. The portable smoke run produced 20 samples per receiver setup,
passed export verification and matched all 18 reference records. Stopping after
the first batch and resuming preserved all 16 cached records. The focused test
suite passed 22 tests covering metadata, data links, sampling, power/noise and
antenna behavior. This used the existing Python environment; installing the
dependencies on a different physical machine has not been tested here.

Review the staged changes with `git diff --cached --stat` and
`git diff --cached`. Staging does not create a commit or push to GitHub.
