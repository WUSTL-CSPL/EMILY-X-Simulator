# What we have done and what comes next

We connected Greg's LTE simulator to a small incident-based localization
dataset. We keep the source power, antennas, traffic and two receiver sites
fixed, and vary the source position. This lets us first ask whether the incident
records contain enough information to locate the source.

The original sampling region produced detections at only 33.525% of positions.
Moving the same-sized region closer to the receivers raised this to 99.675%.
This changes the source population; it does not improve reception in the old
region. The current v3 split holds out geographic blocks for validation and test.

The unified generator reproduced the original 4,000-scene pool and v3 data
tables byte-for-byte in the tested environment. This repository packages that
workflow and pins the upstream simulator. Datasets and externally supplied
captures are separate from the code repository.

Next steps:

1. Train a first location model. Fit preprocessing on train, select settings on
   validation, then evaluate the frozen model on test. Include simple baselines.
2. Check that real EMILY records provide the same fields and that simultaneous
   receiver observations can be grouped reliably. An absent database record can
   mean more than "no signal."
3. Create a separate detectable-harmonic example. Multiple incidents may share
   one transmitter label; source grouping is not yet a learned task here.
4. In new experiments, vary source power, antenna direction and traffic. Add
   transmitter types only after defining their waveforms and detector behavior.

The current data have one transmitter per scene and one transmitter type.
They do not yet support general device classification or mixed-source grouping.
