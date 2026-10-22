# Multi-receiver localization datasets (v7)

We added two receiving sites and generated six datasets using the original omnidirectional receiver model. Every retained scene has at least one incident at every receiver included in that dataset.

| Receivers | Region | Total | Train / validation / test |
| ---: | --- | ---: | --- |
| 3 | 20 × 20 km | 1,000 | 700 / 150 / 150 |
| 3 | 20 × 20 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 3 | 40 × 40 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 4 | 20 × 20 km | 1,000 | 700 / 150 / 150 |
| 4 | 20 × 20 km | 10,000 | 7,000 / 1,500 / 1,500 |
| 4 | 40 × 40 km | 10,000 | 7,000 / 1,500 / 1,500 |

The folders and ZIPs are named `datasets/terrestrial_location_v7_<N>rx_<W>km_<COUNT>`.

## Receiver layout

RX1 and RX2 keep their original coordinates. RX3 is 5 km west and 2 km south of the sampling center; RX4 is 4 km east and 6 km south. This gives a non-collinear layout with site separations of about 9.8–22.7 km. It is a practical reference layout, not an optimized deployment.

| Receiver | Latitude | Longitude |
| --- | ---: | ---: |
| RX1 | 39.52490000 | −114.37332500 |
| RX2 | 39.59551800 | −114.53541800 |
| RX3 | 39.45213924 | −114.51242105 |
| RX4 | 39.41611616 | −114.40788182 |

All receivers use **0 dBi omnidirectional gain, 2 m height and 6 dB noise figure**. The third and fourth receivers use the same hardware parameters as the original receivers. Three-receiver datasets contain RX1, RX2 and RX3.

The original transmitter sector antenna is unchanged. The source remains loaded LTE at 800 MHz, 10 MHz bandwidth, 43 dBm conducted power and 40 m height. Each scene lasts 60 seconds. IQ generation, full P.452/SRTM propagation, receiver effects and the local EMILY detector are reused without a directional receive-antenna adapter.

A separate pilot used only training-region positions: 198 of 200 proposals at 20 km and 117 of 200 at 40 km were detected at all four sites. The pilot used a different seed from the final datasets.

## Sampling and filtering

We keep the original geographic train/validation/test block assignments, including the same center: **39.47016783035665, −114.45433040440209**. The wider square extends the same grid; it does not move held-out core regions.

Within each split, blocks are chosen in proportion to usable area and positions are drawn uniformly inside them. We exclude 250 m inside block edges and positions within 1 km of any of the four candidate sites. The shared four-site exclusion also applies to the three-receiver datasets, to keep their proposal domain comparable.

The generator keeps the first qualifying proposals until each split reaches its requested size. It does not rebalance the retained samples across blocks or change the detector to fill gaps. All candidate masks and rejected observations remain in the local campaign audit files.

**The final datasets describe jointly detectable sources. They do not represent uniform geographic coverage.** Some parts of the 40 km square can be poorly represented or absent after filtering. See each dataset's `selection_report.json` for acceptance and block counts.

## Training and fair comparisons

- A sample groups all receiver incident lists for one source position. Join the three JSONL files by `sample_id`.
- Use `inputs.jsonl` as observations and `labels.jsonl` as location/type targets. All source types are currently `terrestrial_lte`.
- Incident latitude/longitude identify receivers. Ground-truth transmitter coordinates, distances, bearings and path losses must not become model features.
- Intensity is in mW. Preserve null end/duration fields for ongoing incidents.
- The 1,000-scene datasets are subsets of their matching 10,000-scene datasets, including their validation/test sets. Do not concatenate them as independent data.
- Three- and four-receiver versions have different detection conditions, so their final locations are not identical. `common_validation_test.json` gives shared positions for matched evaluation.
- Train on training data, choose settings on validation, and reserve test for the final comparison. Shared positions never move between split roles.

These records support multi-receiver localization from signal observations. They contain no arrival angles or precise arrival times, so they are not direct inputs to classical angle-based triangulation. Having detections at all receivers also does not guarantee a unique location estimate.

## Run

From `emily-location-dataset`, using the existing Python environment in its parent folder:

```bash
../.venv/bin/python generate_multireceiver_dataset_v7.py --config configs/multireceiver_v7.json
../.venv/bin/python analyze_multireceiver_v7.py
```

The generator calls `multireceiver_campaign_v7.py` and imports the original `generate_location_dataset.py`. It also reads the existing v3 dataset for reference settings, block assignments and replay checks. Run it from this repository; the exported ZIP alone is not a standalone simulator installation. Old source files and datasets are unchanged.

[Configuration](../configs/multireceiver_v7.json): sampling seed `20261005`, pilot seed `20261006`, outer split seed `20260930`, 16 workers, batches of 512. The waveform template seed stays at `20260921`.

Generation can resume with `--resume`; it verifies code, configuration, runtime, IQ and terrain before using cached rows. To start a different campaign, copy the config and change both `output_prefix` and `campaign_dir`. The analysis command targets the default campaign and refuses to overwrite its report.

See the [generated comparison report](../datasets/terrestrial_location_v7_comparison/README.md) for downloads, maps, measured acceptance and verification. Local proposal audits are in `runs/multireceiver_v7/`; raw IQ and terrain remain separate dependencies.
