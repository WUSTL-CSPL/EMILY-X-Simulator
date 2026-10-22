# Six datasets with random transmitter power (v8)

## What changed

These are new versions of the six v7 datasets. The receiver sites, omnidirectional hardware, LTE waveform, propagation model and geographic split rules stay the same.

For each proposed source location:

- Draw one conducted TX power uniformly from **33–43 dBm**, about **2–20 W**. The draw is uniform in dBm, not watts. All receivers observe this same source power for the full 60-second scene.
- Draw a separate receiver measurement offset from **N(0, 2²) dB** for each receiver. Add it to the full received spectrogram before EMILY processing. The offset stays fixed during the scene.
- Run the original detector. Keep the scene only if **every included receiver has at least one incident**. Rejected scenes stay in the local audit cache.

The original model already has thermal noise, 0.2 dB pixel jitter, gain drift and 0.8 dB slow shadowing. The new offset adds intensity measurement error. It shifts the measured noise floor along with the signal, so it does not add physical fading or lower signal-to-noise ratio. The detector uses relative contrast and is approximately invariant to this offset. Lowering TX power does reduce signal-to-noise ratio.

The 2 dB value is a modest test setting, not a value fitted to receiver measurements. Gaussian here means Gaussian **in dB**; in linear power it is a multiplicative error. Errors are independent across receivers and scenes, but constant across time and frequency within each scene.

TX power changes actual emission-component powers, including harmonics and spurs. We keep their spectral shapes and relative levels fixed. This models a change in output level; it does not recalculate amplifier nonlinearity at different operating points.

## Dataset sizes

| Receivers | Square | Scenes | Train / validation / test |
| --- | --- | ---: | --- |
| 3 | 20 × 20 km | 1,000 | 700 / 150 / 150 |
| 3 | 20 × 20 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 3 | 40 × 40 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 4 | 20 × 20 km | 1,000 | 700 / 150 / 150 |
| 4 | 20 × 20 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 4 | 40 × 40 km | 10,000 | 7,000 / 1,500 / 1,500 |

The 1,000-scene sets are exact per-split subsets of their 10,000-scene counterparts. Shared positions across v8 variants have the same power, receiver noise and measurement offsets, and always belong to the same split. Do not concatenate these as independent datasets.

## What stays fixed

- RX1: (39.52490000, −114.37332500); RX2: (39.59551800, −114.53541800).
- RX3: (39.45213924, −114.51242105); RX4: (39.41611616, −114.40788182).
- All receivers: 0 dBi omnidirectional gain, 2 m antenna height, 6 dB noise figure.
- Source: terrestrial LTE at 800 MHz, 10 MHz bandwidth, loaded traffic, 40 m tower, original sector antenna and 4° downtilt.
- Full upstream ITU-R P.452-16 propagation with local SRTM terrain; no directional receiver adapter.
- 60-second observation window and the same template schedule.
- The old geographic block roles, a 250 m inside-edge buffer and a 1 km exclusion around all four candidate receiver sites. The exclusion also includes RX4 for three-receiver sets.

Within each split, draw blocks in proportion to their usable area and then sample uniformly inside them. Detection filtering changes the final spatial distribution. The 40 km region extends the existing grid without moving the original held-out blocks.

## Use for training

Each split has three JSONL files, joined by `sample_id`:

- `inputs.jsonl`: receiver IDs and EMILY incidents. Use these as observations. Incident coordinates are **receiver** coordinates.
- `labels.jsonl`: source type and source latitude/longitude. Use these as targets.
- `ground_truth.jsonl`: physical parameters and generation details. Use this for audits, not model inputs.

New ground-truth fields are `transmitter_power_dbm`, `receiver_measurement_offsets_db`, `measurement_offset_sigma_db` and `randomization_seed`. Each link also records its offset, physical noise floor and measured noise floor. The inherited `reference_transmitter` in the source configuration is the 43 dBm baseline; the actual scene power is in ground truth and in its `transmitter.conducted_power_dbm`. Physical component powers and path losses are not shifted by the measurement offset.

Intensity is in mW; convert using `10*log10(value)` for dBm features. Comparing intensities between receivers for the same spectral product can reduce the common TX-power effect. It does not remove independent receiver errors or terrain effects. Keep missing end times for ongoing incidents as missing, not zero.

Random power and receiver errors can make localization harder. All-receiver detection ensures complete receiver inputs, but does not guarantee a unique location. Also, filtering may favor high-power sources at difficult locations. The final analysis reports the power distribution before and after filtering.

## Run

From `emily-location-dataset/`, using the existing parent virtual environment:

```bash
../.venv/bin/python -m unittest discover -s tests -p '*v8.py' -v
../.venv/bin/python generate_multireceiver_dataset_v8.py --pilot
../.venv/bin/python generate_multireceiver_dataset_v8.py --config configs/multireceiver_v8.json
../.venv/bin/python analyze_multireceiver_v8.py
```

The pilot and analysis refuse to overwrite their existing output folders. For an interrupted full run, add `--resume` to the generation command. Resume requires the same code, runtime, data and config. For a different full campaign, copy the config and change `output_prefix` and `campaign_dir`. The pilot and analysis commands currently target their default v8 folders.

Main settings in `configs/multireceiver_v8.json`:

| Setting | Value |
| --- | --- |
| `sampling_seed` | 20261013 |
| `pilot_seed` | 20261014 |
| `extension_split_seed` | 20260930, unchanged |
| `transmitter_power_dbm_min/max` | 33 / 43 |
| `measurement_offset_sigma_db` | 2 |
| `workers` | 16 |
| `batch_size` | 512 |

Position, original receiver noise, TX power and new measurement error use separate random streams. The new position seed gives fresh locations; v8 is not a point-by-point power-only replay of v7. Across v8 variants, common positions reuse the same full four-receiver simulation. Selection follows proposal order, independently of worker completion order.

The generator imports `generate_location_dataset.py` and uses the v3 reference configuration and split manifest. IQ data, SRTM files and the original simulator remain local dependencies. v7 scripts, documents and datasets are preserved. ZIP files contain training data and code snapshots, not all simulator dependencies.

## Outputs and checks

- Six folders and ZIP files: `datasets/terrestrial_location_v8_*`.
- Results and plots: [comparison report](../datasets/terrestrial_location_v8_comparison/README.md).
- Full proposed positions, power, offsets, detection masks and selection flags: `runs/multireceiver_v8/proposals_*.csv`.
- Accepted and rejected simulated scenes: `runs/multireceiver_v8/scene_cache.sqlite`.
- Runtime, input hashes and completion status: `runs/multireceiver_v8/`.

Checks cover actual RF power scaling, zero-change replay against the original, deterministic random streams, geographic separation, all-receiver detection, table alignment, archive contents and preservation of old files. Selected scenes are also simulated again to check exact reproducibility. No model training is part of this generation run.
