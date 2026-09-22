# EMILY location dataset

Generate EMILY-compatible incident records for a first transmitter-location
experiment. Each sample is one terrestrial LTE source observed by two fixed
receivers for 60 seconds. Source coordinates are the training target; simulation
ground truth is kept separate from model inputs.

We use [Greg's EMILY-X simulator](https://github.com/greghell/Incident_simulator)
as a **Git submodule**, pinned to commit
`a6124824f653467173c845c4ff609d4d35d6902f` (v3.2). You do not need write access to
his public repository. Your repository stores a reference to that commit;
Git downloads his code when the submodule is initialized.

[Dataset design](docs/DATASET.md) · [Input data](docs/DATA.md)

## Get the code and data

After this repository is published:

```bash
git clone --recurse-submodules YOUR_REPOSITORY_URL emily-location-dataset
cd emily-location-dataset
```

For an existing checkout, run `git submodule update --init --recursive`.
Use Git to clone it; a GitHub source ZIP does not contain the submodule checkout.

Provide the externally supplied IQ files in `EMILY-X/` and cached terrain in
`srtm_data/`; see [the data guide](docs/DATA.md). To reuse the original workspace
on this machine, run from this new repository:

```bash
python3 scripts/link_local_data.py --source ..
```

This creates local links to the two data folders. Git ignores those links and
their contents. They are a convenience for this machine; collaborators need
their own copies of the data.

## Set up

Use Linux or WSL with Python 3.12 and Git installed:

```bash
bash setup_simulator.sh
.venv/bin/python scripts/check_inputs.py --reference-v3
```

Setup initializes the pinned submodule, creates `.venv`, installs
`requirements.lock.txt`, and checks the dataset inputs. The second command also
checks the simulator, IQ and terrain against the reference hashes.

## Generate the v3 dataset

Run from the repository root. These commands require new output directories.

```bash
.venv/bin/python generate_location_dataset.py \
  --stage train --profile near_south10 --record-format v2 \
  --count 4000 --incident-target 1000 \
  --cells 10 --width-km 20 --min-distance-km 1 \
  --duration 60 --start-utc 2026-09-02T19:00:00+00:00 \
  --seed 20260921 --workers 8 --srtm-dir srtm_data \
  --output datasets/terrestrial_location_v2/campaign

.venv/bin/python split_location_dataset.py \
  --source datasets/terrestrial_location_v2/campaign \
  --output datasets/terrestrial_location_v3 --seed 20260922
```

The first command generates 4,000 locations. The second selects **1,000 training,
200 validation and 200 test samples** from separate geographic blocks. The name
`v2` identifies the existing pool and record format; `v3` identifies its geographic
split. See [all parameters and selection rules](docs/DATASET.md).

An earlier full generation took **155 seconds** on an i9-13900K under WSL2,
using eight workers and local terrain files. That includes startup and output,
but not geographic splitting. Use Bash's `time` before the command to measure
your machine. Repeating a generation command with `--resume` reuses completed
scenes; configuration, code and input checks must still match.

## Verify

```bash
.venv/bin/python scripts/verify_reference.py
.venv/bin/python -m unittest discover -s tests -v
```

The reference check compares the regenerated pool and split with 22 recorded
file hashes. It needs no copy of the old datasets. Source hashes, local paths
and runtime metadata will differ; the incident, label and ground-truth tables
must match exactly in the reference environment.

The historical v1-versus-v2 comparison is skipped when its old experiment files
are absent. Record checks use the newly generated pool when no v1 pilot exists.
Geographic-split integration tests run after generating the pool and split above.
No model is trained or scored on the test set by these commands.
See [the packaging validation record](docs/VALIDATION.md) for checks performed
after moving the workflow into this repository.

## Files

| File or folder | Purpose |
| --- | --- |
| `generate_location_dataset.py` | Unified v1/v2 generation code |
| `generate_location_dataset_v2.py` | Compatibility entry point for old commands |
| `split_location_dataset.py` | Geographic split selection and validation |
| `run_simulator.py` | Optional original terrestrial or satellite demonstration |
| `Incident_simulator/` | Greg's pinned upstream submodule |
| `scripts/` | Input checks, local data links, reference checks and output validation |
| `configs/` | Reference hashes and example load/beam schedules |
| `tests/` | Sampling, record-boundary, resumption and split checks |
| `docs/` | Data requirements, parameters, project scope and GitHub instructions |

IQ, terrain, datasets, run logs, caches and virtual environments stay outside Git.
The existing work directory and historical analyses were left in place.
This repository contains the current generation workflow, not a trained model.
See [project scope and next steps](docs/PROJECT.md).
