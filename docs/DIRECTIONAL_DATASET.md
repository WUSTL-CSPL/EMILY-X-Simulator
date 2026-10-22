# Directional receiver experiment

This experiment changes only the receiving antenna response. It replays the
1,400 scenes already selected for v3: 1,000 train, 200 validation and 200 test.
Source positions, IDs, geographic splits and random seeds stay the same.
It does not regenerate or resample the original 4,000-scene proposal pool.

## Settings

Both antennas point permanently toward the v3 sampling-region center
(39.47016783035665, -114.45433040440209). WGS84 bearings are about 228.934 degrees
for RX1 and 153.367 degrees for RX2, measured clockwise from north.

The horizontal gain is `-min(12 * (off_axis / 120)^2, 20)` dBi, with the signed
off-axis angle wrapped into [-180, 180). The peak is 0 dBi. At 60 degrees off
axis the gain is -3 dBi; at 90 degrees it is -6.75 dBi; behind the antenna it
is -20 dBi. The response is smooth, with no hard cutoff. There is no vertical
or frequency dependence. These are controlled experiment settings, not a
measured antenna specification.

Receiver location, 2 m height, 6 dB noise figure, noise/background draws, source
emissions, terrain propagation and detector settings stay unchanged. Because
the source is stationary, the new wrapper can calculate one effective gain
for each source-to-receiver link and pass it into the existing simulator.
The source signal is attenuated before receiver noise is added and incidents
are detected. Existing incidents are not edited or scaled after detection.

## Run

The new files are `generate_directional_location_dataset.py` and
`configs/directional_receivers_v4.json`. Neither the old generator nor Greg's
simulator needs editing. The wrapper imports the existing generator, so keep
both scripts and the pinned simulator submodule available.

From this repository, with the original v3 dataset, IQ files, terrain and
Python environment available:

```bash
.venv/bin/python generate_directional_location_dataset.py \
  --config configs/directional_receivers_v4.json
```

On the current machine the environment is in the parent workspace, so use
`../.venv/bin/python` instead. Config paths are relative to this repository.
The example uses eight workers. Edit the **new** config for a different
beamwidth or maximum attenuation; this experiment requires the peak to remain
0 dBi. A new output directory is required for each run:

```bash
../.venv/bin/python generate_directional_location_dataset.py \
  --config configs/directional_receivers_v4.json \
  --output datasets/another_directional_run
```

`--output` is relative to the shell's current directory. The script refuses
to overwrite an existing dataset or ZIP. It does not resume partial runs.

## Output and checks

The default output is `datasets/terrestrial_location_v4_directional/` and a
ZIP beside it. Each split has the same three JSONL tables as v3. Every original
sample remains in its original split and order, even if both receivers now
have empty incident lists. Keep those scenes when measuring coverage. If you
evaluate localization only on detected scenes, report that selection and its
coverage explicitly; do not replace empty scenes with new locations.

Labels are byte-identical to v3. Input fields are unchanged. Ground truth adds:

- `receiver_antenna_patterns`: fixed antenna settings, including pointing.
- `links.RX*.rx_antenna_response`: source bearing, off-axis angle and effective
  gain for this particular scene. These depend on the true source position
  and must not be used as model inputs.

`receivers[].antenna_gain_dbi` remains the fixed boresight gain of 0 dBi;
the scene-dependent effective gain is recorded separately as described above.
Known antenna settings can be inference metadata in a later model, provided
the same information is available at deployment.

Before the full run, one original scene from each split must reproduce exactly,
and the wrapper with zero attenuation must reproduce those observations. Every
new scene is checked against its original for unchanged geometry, propagation,
noise and source emissions, and for the expected source power attenuation.
The script also checks reference terrain/IQ and verifies original files are
unchanged. `comparison.csv`, `summary.json` and the generated `README.md`
report detection coverage. `verification.json` records checks and table hashes.

Run the focused tests from this repository:

```bash
../.venv/bin/python -m unittest discover -s tests -p 'test_directional_dataset.py' -v
```

This experiment tests antenna directionality. It does not measure arrival
angle, and it does not establish a localization improvement until a model is
trained and evaluated. No extra gain is introduced, so coverage may decrease.
