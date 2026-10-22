# V5: a manufacturer-based directional receiver model

V5 uses the POYNTING LPDA-92 gain curve and antenna patterns. It replays the
same 1,400 source scenes as v3: 1,000 train, 200 validation and 200 test.
Positions, sample IDs, labels, noise seeds and geographic splits stay the same.
All scenes are retained, including any that lose detection at both receivers.

## What changed

V4 had a 120-degree horizontal pattern with a 0 dBi peak. V5 uses a digitized
gain-versus-frequency curve and horizontal/vertical cuts from the
[manufacturer's datasheet](https://drive.google.com/uc?export=download&id=1pup5J92OWVpsRE8j_v-te_hXMwyzA138),
linked from the [LPDA-92 product page](https://poynting.tech/antennas/lpda-92/).
The source is revision 7.0, February 2023; its SHA256 is recorded in the table.

- Boresight gain at 800 MHz is about **10.84 dBi**, before cable loss.
- RX1 points at **228.934 degrees**; RX2 at **153.367 degrees**. These bearings
  point toward the fixed v3 sampling center, not each sample's source.
- Both antennas have a fixed elevation of **0 degrees**.
- Gain varies with signal frequency and arrival direction. It can increase
  forward reception and reduce reception off to the sides or behind.
- Gain is applied before noise is added and incidents are detected. The
  original transmitter, receiver locations/heights/noise figures, noise draws,
  terrain propagation and detection settings are retained.

## What the manufacturer data do and do not provide

The public source contains plots, not raw chamber measurement tables. The new
digitization script extracts the plot colors and scales, samples angles every
5 degrees and interpolates gaps. Overlapping main-lobe curves sometimes hide
one another; those missing angles use the visible main-lobe envelope from the
same frequency panel. Other missing angles are interpolated. The table records
the gap/reconstruction counts. These curves have no claimed measurement-level
accuracy; they are more specific than a generic parabola but still approximate.

The two principal-plane cuts are combined as
`gain(f, az, el) = peak(f) + max(H(f, az) + V(f, el), -50 dB)`.
This is a separable 3D approximation, not the manufacturer's full 3D response.
Horizontal angles use a top-view convention; geographic offsets are clockwise
and the PDF polar angles counter-clockwise. Positive vertical angles point up.

Frequency interpolation is linear in dB. In particular, the 1600 MHz harmonic
uses interpolation across the published 870-1710 MHz pattern gap; it is not a
measurement at 1600 MHz. The source's active spectral bins are within the
digitized 690-2500 MHz range. The code rejects active signal frequencies outside
that range rather than silently extending the pattern. The wider noise-only
observation grid remains unchanged.

The default represents gain at the antenna reference plane: additional cable
loss is 0 dB, and matched polarization is assumed. It does not model the
LPDA-92's supplied cable, mounting structures, polarization mismatch, or
directional environmental noise. Geometric arrival angles approximate the
path direction; reflected/scattered arrivals are not resolved separately.

The original P.452 calculation is retained, including zero internal `G_t/G_r`
and a separate antenna gain in the link budget. V5 does not implement antenna
gain coupling inside the troposcatter term. These limitations should remain
visible when interpreting a localization result.

## New files and running

All implementation files are new. Existing scripts, Greg's simulator and
v3/v4 datasets are not edited.

| File | Purpose |
| --- | --- |
| `generate_real_antenna_dataset_v5.py` | Replay existing v3 splits and export v5 |
| `receiver_antenna_v5.py` | Read the antenna table and apply gain per frequency |
| `configs/real_antenna_receivers_v5.json` | Antenna settings, paths and worker count |
| `configs/antennas/lpda92_digitized_v5.json` | Numeric curves, source hash and extraction audit |
| `scripts/digitize_lpda92_v5.py` | Reproduce the table from the pinned PDF |
| `tests/test_real_antenna_v5.py` | Gain, interpolation, noise and upstream-copy checks |

From the `emily-location-dataset` repository on the current machine:

```bash
../.venv/bin/python generate_real_antenna_dataset_v5.py \
  --config configs/real_antenna_receivers_v5.json
```

The script imports the existing `generate_location_dataset.py`; it does not
require the v4 generator to perform the simulation, although it fingerprints
that file to check preservation. Keep Greg's pinned submodule, IQ and terrain
available as for v3. A fresh checkout can use its own `.venv/bin/python`.

Config paths are relative to the repository. `--output PATH` overrides the
destination relative to the shell's current directory. The default destination
is `datasets/terrestrial_location_v5_lpda92/`, with a ZIP alongside it.
Existing destinations are rejected. There is no partial-run resume.

To rerun the focused tests:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_real_antenna_v5.py' -v
```

To reproduce the antenna table, use a **new** output path (requires Poppler's
`pdfimages`, NumPy and Pillow). `--pdf PATH` can supply an already downloaded
copy; otherwise the script downloads the public manufacturer PDF:

```bash
../.venv/bin/python scripts/digitize_lpda92_v5.py --output /tmp/lpda92_check.json
```

## Data and validation

Input and label schemas are unchanged. Labels are checked byte-for-byte
against v3. In ground truth, `receivers[].antenna_gain_dbi` now reports the
boresight gain at `antenna_gain_reference_frequency_hz` (800 MHz).
`receiver_antenna_patterns` holds fixed settings; `links.RX*.rx_antenna_response`
holds arrival angles and the effective gains at 800 MHz and component centers.
The latter fields depend on source truth and must not be model inputs.

Before generation, one original scene from each split must reproduce exactly.
The new spectrum implementation with uniform 0 dBi must reproduce those
observations too. Every generated scene is checked against the original for
unchanged geometry, propagation, noise and emissions, and the expected gain
change in component powers. The ZIP is checked after packaging.

Read the generated `README.md`, `summary.json` and `comparison.csv` for coverage
results, and `verification.json` for checks and hashes. Coverage changes alone
do not show that localization improved. No model is trained by this script.
