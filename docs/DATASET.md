# Dataset settings and selection

## What one sample means

One source at one location transmits loaded LTE for 60 seconds. RX1 and RX2
observe it simultaneously. Both receiver groups, including empty groups, stay
together. A sample can contain several incidents from the same transmitter.

Each split has three JSONL files joined by `sample_id`:

- `inputs.jsonl`: receiver groups and EMILY-compatible incident fields.
- `labels.jsonl`: `terrestrial_lte` and the source latitude/longitude.
- `ground_truth.jsonl`: source settings, propagation and other audit information.

Use incident fields as inputs and coordinates as targets. Do not feed the join
ID or ground truth into the model. Receiver coordinates in incidents describe
the observation site, not the transmitter.

## Generation parameters

The full command is in [the README](../README.md#generate-the-v3-dataset).

| Parameter | Value and meaning |
| --- | --- |
| `--stage` | `train`: random positions within each sampling cell |
| `--profile` | `near_south10`: region centered 10 km south of the geodesic midpoint of the two receivers |
| `--record-format` | `v2`: preserve the existing schema and `v2_train_...` IDs |
| `--count` | `4000`: simulate all proposals, including those with no detections |
| `--cells`, `--width-km` | `10`, `20`: 100 cells, each 2 × 2 km; 40 random positions per cell |
| `--min-distance-km` | `1`: resample within the same cell if too close to either receiver |
| `--duration` | `60` seconds |
| `--start-utc` | `2026-09-02T19:00:00+00:00`, identical for every scene |
| `--seed` | `20260921`: fix locations, shuffle order, template schedule and receiver randomness |
| `--workers` | `8` parallel processes |
| `--srtm-dir` | `srtm_data`: cached elevation tiles |
| `--incident-target` | `1000`: recreate the historical comparison sample list used by the split's exclusion rule |
| `--output` | `datasets/terrestrial_location_v2/campaign`: the complete candidate pool |

The center is latitude **39.47016783035665**, longitude **−114.45433040440209**.
Use the named profile for exact reproduction rather than rounded coordinates.

`incident_training/` is an intermediate historical selection, not the final v3
training set. The same list was previously used to compare models, so validation
and test must exclude its IDs when reproducing v3.

## Fixed physical settings

| Part | Settings |
| --- | --- |
| Source | Terrestrial LTE, 800 MHz center, 10 MHz bandwidth, 43 dBm conducted power |
| TX antenna | Height 40 m above ground, gain 17 dBi, horizontal beamwidth 65°, azimuth 10°, downtilt 4° |
| RX1 | Latitude 39.5249, longitude −114.373325 |
| RX2 | Latitude 39.595518, longitude −114.535418 |
| Receivers | Height 2 m above ground, gain 0 dBi, noise figure 6 dB, extra path loss 0 dB |
| IQ | Loaded capture, 11.52 MS/s, five 100 ms templates from 0.5 seconds |
| Terrain and loss | SRTM + ITU-R P.452-16, 100 m profile spacing, 50% time percentage, 2 MHz loss anchors |
| Spectrum | 200–2500 MHz, 70 kHz channels, 140 kHz resolution bandwidth, 1-second samples |
| Receiver variations | Noise jitter 0.20 dB, bandpass ripple 0.35 dB, gain drift 0.20 dB, shadowing sigma 0.8 dB and correlation 30 s |
| RF effects | Upstream filters, amplifier distortion, leakage, spurs and harmonics; harmonics at −60/−65 dBc |
| Detector | Upstream threshold 8, local windows ±40 MHz about 800, 1600 and 2400 MHz |

The pinned simulator commit fixes the remaining model constants. Harmonics are
enabled, but none produced detected incidents in the checked reference runs.

## Geographic split

The splitter uses seed **20260922**, independent of the generation seed.

1. Divide the same square into **25 blocks of 4 × 4 km**. These are larger than
   the 100 generation cells.
2. Randomly assign whole blocks: 17 training, 4 validation, 4 test.
3. Exclude points within 250 m of any block edge, and scenes with no incidents.
4. For validation and test, also exclude previously evaluated sample IDs.
5. Randomly select 58 or 59 samples per training block and 50 per validation/test
   block, giving **1,000 / 200 / 200**.

Block roles are assigned without looking at detection strengths or model scores.
The buffer gives at least 500 m between sets. The measured minimum was about
608 m. The block size and buffer are practical choices, not proof that terrain
effects become statistically independent beyond that distance.

Reference pool: 3,987 detected scenes, including 3,950 detected at both receivers;
13 empty scenes. The 2,600 unselected scenes remain in the pool. Final train,
validation and test have 995, 195 and 200 scenes detected at both receivers.

## Reproduction and changes

Matching seeds alone is insufficient: keep IQ, terrain, time, physics, detector,
software versions and random draw order fixed. Input hashes and reference output
hashes are tracked under `configs/`. Runtime paths and code hashes are provenance
and are expected to differ between workspaces.

Use `--stage pilot --profile near_south10` for a 100-position preview, or
`--stage survey` to compare the seven original region/antenna presets.
The original v1 behavior uses `--profile original --record-format v1
--min-distance-km 0`.

Custom runs support `--profile custom --center-lat LAT --center-lon LON`,
`--tx-height-m`, `--rx-height-m`, `--downtilt-deg`, `--width-km`, `--cells` and
`--duration`. Counts must be divisible by `cells²`. The current splitter is
specifically configured for a 20 km square and the historical exclusion rule;
review that design before applying it to a different experiment.
