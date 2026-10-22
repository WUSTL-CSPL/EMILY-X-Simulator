# Two 10,000-scene datasets

These datasets let us compare two ways of adding data: sample the current region more densely, or cover a wider region with the same sample budget. We have not trained a model on them yet.

| Dataset | Region | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| `terrestrial_location_v6_20km` | 20 × 20 km | 7,000 | 1,500 | 1,500 |
| `terrestrial_location_v6_40km` | 40 × 40 km | 7,000 | 1,500 | 1,500 |

A scene is one stationary LTE source observed by both receivers for 60 seconds. Counts refer to scenes, not individual incidents.

## What stayed the same

Both datasets use the v5 LPDA-92 antenna model, with the same two sites and fixed antenna directions. The center remains **39.47016783035665, −114.45433040440209**. The LTE signal remains 800 MHz, 10 MHz bandwidth, 43 dBm conducted power, and loaded traffic. The transmitter stays at 40 m and receivers at 2 m. IQ, detector settings and P.452/SRTM propagation are unchanged.

The antenna model uses digitized manufacturer curves. It is an approximation, not a calibrated measurement of the receiver installations.

## How we sample

- Keep the original 4 km geographic grid and its 25 core block assignments.
- Give added outer blocks their own fixed train/validation/test assignments before simulation.
- Exclude 250 m inside each block edge, leaving at least 500 m between different splits. Keep sources at least 1 km from each receiver.
- Allocate each split's sample budget by usable rectangular block area, then draw uniformly inside each block. Resample only points inside the receiver exclusion circles.
- Retain all generated scenes, including those with no incidents. This differs from the historical v3/v5 selection, which contained only detected scenes.

Keeping the old grid while doubling the centered square leaves half-width blocks along the outside boundary. The 40 km version therefore has **121 rectangles**, including 2 km edge strips. Shifting the grid to make 100 equal blocks would change the old held-out regions.

Both datasets share **412 validation scenes and 436 test scenes**, with identical observations and labels. The wider training set has 1,734 core scenes and 5,266 outer scenes. Its core training scenes are also present in the smaller-region dataset. No scene changes split.

## How to use them

Each split contains `inputs.jsonl`, `labels.jsonl` and `ground_truth.jsonl`, joined by `sample_id`. Train with incident inputs and coordinate labels. Keep ground truth and IDs out of the features.

Train separate models on the two datasets. Compare them on the shared core validation IDs in `common_evaluation.json`. Use the shared test IDs only for the final comparison, and report outer-region performance separately. Fit preprocessing on each training set only. Do not concatenate the datasets without removing duplicates.

Report detection coverage alongside localization error on detected scenes. Empty inputs do not identify a unique location; they also do not mean that the source was absent. Comparing whole-domain errors alone would mix the effect of different training data with the effect of different evaluation regions.

## Run

From `emily-location-dataset`, using the existing environment in the parent directory:

```bash
../.venv/bin/python generate_location_comparison_v6.py --config configs/location_comparison_v6.json
../.venv/bin/python analyze_location_comparison_v6.py
```

Parameters are in [location_comparison_v6.json](../configs/location_comparison_v6.json): sampling seed `20260929`, outer split seed `20260930`, eight workers. The waveform template schedule retains the v5 seed `20260921`. Large IQ and terrain files remain local dependencies.

The generator creates new dataset folders and ZIPs under `datasets/`; it refuses to overwrite existing outputs. To regenerate, copy the config and use new output paths. The analysis script targets the default v6 directories and also refuses to overwrite its report directory.

See the [generated comparison report](../datasets/terrestrial_location_v6_comparison/README.md) for coverage, maps, timings in the dataset summaries, and verification results. Existing generators, documents and datasets were left unchanged.
