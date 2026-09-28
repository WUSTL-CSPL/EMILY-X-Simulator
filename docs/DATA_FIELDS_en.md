# The three JSONL files: fields and training guide

This guide describes `terrestrial_location_v3`. Field meanings were checked
against the dataset and generator source. [中文版](DATA_FIELDS_zh.md)

JSONL stores one JSON object per line. Here, each line represents one source
location and its observations at both receivers. A line can contain several
incidents. Join the three files by `sample_id`, rather than relying on row order.

| File | Contents | Role in location training |
| --- | --- | --- |
| `inputs.jsonl` | What the two receivers detected | Build model inputs X |
| `labels.jsonl` | The source's true location and type | Build training targets y |
| `ground_truth.jsonl` | Source settings, propagation and random parameters | Check the simulation and investigate errors; exclude from the first model's inputs |

## 1. inputs.jsonl

### Outer structure

| Field | Meaning |
| --- | --- |
| `sample_id` | Join key, such as `v2_train_003221`. Do not use it as a feature. The word train refers to the original generation stage; the containing folder determines the final split. |
| `receivers` | Observations from the two receivers. |
| `receivers[].receiver_id` | `RX1` or `RX2`. |
| `receivers[].incidents` | Events detected at this receiver during the window. An empty list `[]` means the simulated receiver was observing normally but detected no events. |

One source can produce several incidents or be detected by only one receiver.
Keep both receivers and all their incidents together as one training sample,
with one source label.

### Fields in each incident

These fields appear inside `receivers[].incidents[]`.

| Field | Meaning and units |
| --- | --- |
| `data_type` | Observation type. Always `spectrometer` in this dataset. |
| `t_start` | Recorded start time in UTC. If the event is already present at the start of the window, this is the window start, not the actual transmission start. |
| `t_mid` | Midpoint of the first and last observed event indices, rounded to a sampled time. It is not a power-weighted center. |
| `t_end` | Recorded end time in UTC; `null` when `ongoing=true`. |
| `t_start_offset_sec` | Start time in seconds relative to the observation start. Zero if the event is already present at the first sample. |
| `t_mid_offset_sec` | Midpoint time in seconds relative to the observation start. |
| `t_end_offset_sec` | End time in seconds relative to the observation start; `null` when the end was not observed. |
| `duration_sec` | End minus start when both are known. May be `null` for events reaching a window boundary. Unknown does not mean zero. |
| `ongoing` | Whether the event reaches the final observation sample. True means its end was not observed, not that it continues forever. |
| `time_structure` | The current extractor uses `continuous` for events touching either window boundary and `bounded` for events contained within the window. This is not a separate classification of continuous-wave versus pulsed modulation. |
| `f_low_hz` | Lowest frequency sample occupied by the detected event, in Hz. |
| `f_high_hz` | Highest frequency sample occupied by the detected event, in Hz. |
| `f_center_hz` | Mean of the two frequency boundaries. It is not a precise estimate of the transmitted carrier. |
| `bw_hz` | `f_high_hz - f_low_hz`: the detected frequency span, which need not equal the configured 10 MHz channel bandwidth. |
| `intensity_kind` | Intensity unit; currently `mW`. |
| `intensity_value` | Highest received spectral power among the detection-mask pixels assigned to the event, in mW. It is neither power integrated across the signal bandwidth nor transmitter power. The spectrum uses a 140 kHz resolution bandwidth (RBW). |
| `lat`, `lon` | Receiver latitude and longitude, in degrees. These are not source coordinates. |
| `altitude` | Height field in the receiver location record. Always null here; do not interpret it as zero meters or as the 2 m antenna height above ground. |
| `file_freq_low_hz` | Lowest frequency sample in the full spectrum: 200,000,000 Hz. |
| `file_freq_high_hz` | Highest frequency sample in the full spectrum: 2,499,990,000 Hz. The grid does not land exactly on 2.5 GHz. |
| `file_freq_resolution_hz` | Frequency sample spacing: 70,000 Hz. This differs from the 140 kHz RBW. |
| `file_time_resolution_sec` | Time sample spacing: 1 second. |
| `observation_start` | First sampled time in the full observation window. |
| `observation_end` | Last sampled time in the full observation window. |
| `observation_duration_sec` | Last sampled time minus first sampled time: 59 seconds here. |
| `n_time_samples` | Number of time samples in the full spectrum: 60. |
| `n_freq_channels` | Number of frequency channels in the full spectrum: 32,858. |
| `n_pixels` | Number of time-frequency pixels assigned to the event by the detection mask, including the effects of the detector's mask expansion and grouping. |
| `n_time_pixels` | Number of distinct time samples containing event pixels. There may be gaps between the first and last samples. |
| `n_freq_pixels` | Number of distinct frequency samples containing event pixels. The event need not fill its entire frequency span. |

### Three details worth checking

**The 60-second simulation has a 59-second timestamp span.** Its 60 samples
are at 0, 1, ..., 59 seconds. `observation_duration_sec` is defined as last minus
first; no sample is missing.

**Null is not zero.** The training set contains 2,179 incidents. Of these,
2,170 have a null `duration_sec`, and 2,150 have `ongoing=true`. Do not discard
most of the data because the full event duration is unknown. A first model can
use `n_time_pixels / n_time_samples` to describe detection coverage within the
window, together with `ongoing`. This ratio does not give the event's full
physical duration. If you include nullable fields, also provide a missing-value
mask: a flag telling the model which values were unavailable.

**Convert power to dBm.** For the positive mW values in this dataset, use
`10 * log10(intensity_value)`. For example, RX1 in the first training sample has
about `2.06e-9 mW`, or `−86.9 dBm`. Keep the power definition and receiver
processing consistent when comparing receivers.

## 2. labels.jsonl

| Field | Meaning and use |
| --- | --- |
| `sample_id` | Join key matching the input record. Exclude it from model features. |
| `transmitter_type` | Always `terrestrial_lte` here. Retained for later experiments with more types. |
| `latitude_deg` | True source latitude, in degrees. A location target. |
| `longitude_deg` | True source longitude, in degrees. A location target. |

For example, source `v2_train_003221` is near `(39.41999176, −114.46587764)`.
This is different from the receiver coordinates in its incidents.

For training, convert latitude/longitude to local `(x_km, y_km)` coordinates:
distance east and north of the region center. Predict these two values, then
convert back to latitude/longitude. Use the generator's WGS84 local azimuthal
equidistant (`aeqd`) projection, centered at latitude `39.47016783035665` and
longitude `−114.45433040440209`. These targets can be calculated from labels alone.

## 3. ground_truth.jsonl

These fields explain why a simulated observation looks the way it does. At
inference time, the source distance, direction and path loss are usually
unknown, so they cannot be inputs to the incident-only location model. True
coordinates, including a coordinate transformation, are valid training targets;
they must not be mixed into X.

### Outer fields

| Field | Meaning |
| --- | --- |
| `sample_id` | Sample join key. |
| `x_m`, `y_m` | Local source coordinates relative to the region center, in meters. Positive x is east; positive y is north. |
| `latitude_deg`, `longitude_deg` | True source coordinates, matching labels. |
| `cell_x`, `cell_y` | Indices in the 10 × 10 generation grid, from 0 to 9. These are not IDs for the 25 larger split blocks. |
| `receiver_seeds` | Random seeds for RX1 and RX2, used to reproduce noise and other random variations. |
| `geometry_replacements` | Number of times the location was redrawn within its cell because it was less than 1 km from a receiver. |
| `source_present` | Whether a source exists in the simulation. Always true here; an existing source can still go undetected. |
| `transmitter` | Source configuration, detailed below. |
| `receivers` | Configuration of the two receivers. |
| `links` | Geometry, propagation and signal-component truth for RX1 and RX2. |
| `profile` | Region and antenna preset used for generation. |

### transmitter fields

| Field | Meaning |
| --- | --- |
| `name` | Source name; currently `LTE_cell_tower`. |
| `lat_deg`, `lon_deg` | True transmitter latitude and longitude. |
| `height_agl_m` | Transmit antenna height above ground level (AGL): 40 m. |
| `conducted_power_dbm` | Total main LTE signal power before the antenna: 43 dBm, excluding antenna gain. |
| `center_frequency_hz` | Configured LTE carrier frequency: 800 MHz. |
| `channel_bandwidth_hz` | Configured LTE bandwidth: 10 MHz. |
| `max_gain_dbi` | Peak antenna gain: 17 dBi. |
| `horizontal_hpbw_deg` | Horizontal half-power beamwidth: 65 degrees. |
| `sector_azimuth_deg` | Antenna pointing direction, clockwise from north: 10 degrees. |
| `electrical_downtilt_deg` | Electrical downtilt: 4 degrees. |

### receivers[] fields

| Field | Meaning |
| --- | --- |
| `name` | RX1 or RX2. |
| `lat_deg`, `lon_deg` | Receiver latitude and longitude. |
| `height_agl_m` | Receive antenna height above ground: 2 m. |
| `antenna_gain_dbi` | Receive antenna gain: 0 dBi. |
| `noise_figure_db` | Receiver noise figure: 6 dB. |
| `excess_path_loss_db` | Additional imposed loss: 0 dB. |

Receiver coordinates are known site information at deployment. If a later model
uses them, read them from observations or a known site configuration, rather
than passing the entire ground-truth record to the model.

### links.RX1 / links.RX2 fields

| Field | Meaning |
| --- | --- |
| `distance_m`, `distance_km` | Ground-path distance from source to receiver, in meters and kilometers. |
| `bearing_tx_to_rx_deg` | Bearing toward the receiver, measured at the transmitter. |
| `bearing_rx_to_tx_deg` | Bearing toward the transmitter, measured at the receiver. This directly reveals the target direction. |
| `elevation_tx_to_rx_deg` | Elevation angle from the transmit antenna toward the receive antenna, including terrain, antenna heights and the model's curvature correction. |
| `noise_floor_per_rbw_dbm` | Model noise floor per resolution bandwidth: about −116.51 dBm. |
| `propagation_model` | Currently `ITU-R P.452-16 + SRTM`. |
| `center_losses_db` | Losses in dB at selected frequencies. Keys are frequencies in Hz stored as strings, such as `"800000000"`. Includes 800/1600/2400 MHz and the two reference-clock spur frequencies. |
| `tx_ground_elevation_m` | Ground elevation at the transmitter. |
| `rx_ground_elevation_m` | Ground elevation at the receiver. |
| `midpoint_ground_elevation_m` | Ground elevation at the midpoint sample of the terrain profile. |
| `shadowing_db_t` | Slow shadowing corrections at 60 time samples, in dB. This implementation adds them to signal power in dB, so positive values increase received power. |
| `components` | Link information for every modeled signal component, including undetected components. |
| `event_censoring` | Whether detected events reach the observation boundaries. |

Distance, direction and path loss depend on the true source location. Keep them
out of model features. Use them afterward to investigate whether errors are
associated with terrain blockage, weak signals or other conditions.

### links.*.components[] fields

| Field | Meaning |
| --- | --- |
| `component` | Component name: `LTE_fundamental`, `harmonic_2`, `harmonic_3`, `carrier_lo_leakage`, `reference_clock_minus` or `reference_clock_plus`. |
| `kind` | Main signal with out-of-band leakage, harmonic, or spurious emission. |
| `center_hz` | Configured center frequency of the component. |
| `tx_antenna_gain_db` | Transmit antenna gain used for this component along this path. |
| `path_loss_center_db` | Propagation loss at the component's center frequency. |
| `median_tx_total_power_dbm` | Median total transmitted power of this component during the window. |
| `median_rx_total_component_dbm` | Median received power calculated from component total power, center-frequency loss, antenna gains and shadowing. This is a component link-budget value, not an incident's peak spectral power. |

These records do not provide a one-to-one incident-to-component mapping.
Components can merge into one detection or remain undetected, so do not assume
`components[0]` corresponds to `incidents[0]`. What is known here is that each
sample's observations come from one simulated source.

### links.*.event_censoring[] fields

| Field | Meaning |
| --- | --- |
| `event_id` | Event number for this receiver's detection run. It is not a source ID across samples, and it is not retained in inputs. |
| `left_censored` | The event is already detected at the first sample, so its true start is unknown. |
| `right_censored` | The event is still detected at the last sample, so its true end is unknown. |

### profile fields

| Field | Meaning |
| --- | --- |
| `region` | How the region center is constructed. Currently `midpoint`, using the geodesic midpoint between the receivers as a reference. |
| `south_km` | Distance south of that midpoint: 10 km. |
| `tx_height_m` | Preset transmit antenna height above ground: 40 m. |
| `rx_height_m` | Preset receive antenna height above ground: 2 m. |
| `downtilt_deg` | Preset transmit antenna downtilt: 4 degrees. |

## 4. Is this data suitable for training?

**Yes, for an initial location experiment with two fixed receivers, one LTE
configuration and a defined region.** There are clear location targets,
separate observations and truth, and geographic train/validation/test splits.
The scope is still limited:

- There is only one type, `terrestrial_lte`, so this cannot test classification
  across different transmitter types.
- Power, antennas, frequency, traffic and receiver sites are fixed. All locations
  share the same short IQ template bank. Performance on real observations,
  new sites or different devices has not been established.
- Power and spectrum statistics from two receivers do not guarantee a unique
  location. Different positions can produce similar observations; evaluation
  must establish the achievable accuracy.
- The final splits retain only scenes detected by at least one receiver.
- Incidents from the same source are already grouped. Forming those groups when
  multiple real sources overlap is a separate problem.
- The 1,000 training scenes are not 2,179 independent training samples. Keep each
  scene's incidents together.

## 5. A first training workflow

### A. Join the records

Read inputs and labels separately for train, validation and test. Join by
`sample_id`, checking that IDs are unique and both files have exactly the same
ID set. Keep the existing splits; do not combine them and split randomly again.

### B. Build fixed-length inputs

Start with per-receiver event summaries for a simple baseline. The following
is a starting proposal, not a model already shown to be optimal.

Extract ten values per receiver:

1. Whether any incident was detected.
2. Incident count.
3. Maximum of the incident peak powers, in dBm.
4. Mean of the incident peak powers, in dBm. This is a mean of peaks, not mean
   total received power.
5. Mean `bw_hz`, converted to MHz.
6. Mean `f_center_hz`, converted to MHz.
7. Mean `n_time_pixels / n_time_samples`.
8. Mean `n_freq_pixels`.
9. Mean `n_pixels`.
10. Fraction of incidents with `ongoing=true`.

Place RX1 first and RX2 second to form 20 input features. You can also add a
flag indicating detection at both receivers, and a difference between their
peak dBm values, calculated only when both have detections.

For an empty incident list, retain `has_detection=0` and `incident_count=0`.
Fill the other values using a fixed rule, such as zero after standardization,
while keeping the detection flag. An absent detection does not mean the true
received power was zero mW. Fit scaling for power and other available-value
statistics using only observed values in the training set.

For the first model, omit IDs, fixed absolute dates, constant file settings,
the entirely missing altitude, and mostly unknown durations. Receiver
coordinates are also constant within each fixed RX slot. Add known site
coordinates explicitly when extending the model to changing receiver sites.

Summaries lose details such as which frequency had which intensity. Once a
baseline works, compare summaries by frequency band, or encode individual
incidents and combine them as a set within each receiver.

### C. Choose targets and a model

Calculate `(x_km, y_km)` from labels as two regression targets. Compare simple
methods, such as predicting a fixed position or using nearest neighbors, before
trying a small multilayer perceptron (MLP), for example `20 → 32 → 32 → 2`.
Choose layer sizes, regularization and stopping time using validation data.
Coordinate mean squared error is one training loss; convert predictions back
to latitude/longitude afterward.

Accuracy on the single type does not demonstrate device recognition. Train the
type output once the data contain multiple classes.

### D. Evaluate and run inference

- **Train: 1,000 samples.** Fit the model, input scaling and any target scaling.
- **Validation: 200 samples.** Select the model, tune settings and stop training.
- **Test: 200 samples.** Evaluate the final frozen model; do not repeatedly adjust
  the design using these results.
- Report distance from predicted to true position: median error, P90, and the
  fraction within 1 km or 5 km.
- At inference, provide receiver incidents in the same structure, apply the
  saved preprocessing, predict `(x, y)`, then convert to latitude/longitude.
  Labels and ground truth are not needed.

Fit preprocessing on training data only, then apply the same transformation to
validation and test. This follows the
[scikit-learn guidance on avoiding data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

This guide describes data semantics and training-set statistics. No model was
trained or selected using the test set for this explanation.
