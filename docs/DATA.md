# Input data

Code comes from Git. The IQ captures and terrain come from separate data files.
They are not included in this repository or downloaded by setup.

## Dataset generation

Required layout:

```text
EMILY-X/
  srsran_loaded.cf32
  srsran_loaded.cf32.json
srtm_data/
  J11/
    ... .hgt terrain files ...
```

Use Greg's supplied data bundle for IQ. The `.cf32` file contains complex radio
samples; the JSON sidecar describes the sample rate, sample count and format.
The generator uses the loaded LTE state only. It builds five spectrum templates
from the half-second capture, then reuses them through each 60-second scene.

SRTM files are elevation-map tiles. Every new source position gets a new terrain
profile to each receiver and a new P.452 loss calculation. The local tile cache
saves downloading maps again; it does not reuse an old source's path loss.

To reproduce v3, use the tiles listed in `configs/reference_v3_inputs.json`.
Copy the original `srtm_data/` directory, preserving its subdirectories, or use
`scripts/link_local_data.py` on a machine that has the previous workspace.
The dataset generator disables terrain downloads. For a different region,
obtain tiles covering the new source-to-receiver paths before generation.

Check reference data:

```bash
.venv/bin/python scripts/check_inputs.py --mode dataset --reference-v3
```

The reference hash list is small and tracked in Git. It identifies required
files; it is not a download service. The supplied IQ bundle and original terrain
cache still need to be shared separately with collaborators.

## Original simulator demonstrations

The optional `run_simulator.py` wrapper needs more IQ states:

| Mode | Required captures, each with its `.json` sidecar |
| --- | --- |
| `terrestrial` | `srsran_idle`, `srsran_light`, `srsran_medium`, `srsran_loaded` (`.cf32`) |
| `leo` | The corresponding four `srsran_d2c_*` captures |

```bash
.venv/bin/python scripts/check_inputs.py --mode terrestrial
.venv/bin/python run_simulator.py terrestrial --duration 60
```

The upstream terrestrial demonstration can fetch missing SRTM tiles when its
configured server is available. That differs from the dataset generator's
offline terrain policy. A single demonstration path does not necessarily fetch
all the tiles needed for a 4,000-position campaign.

The satellite example uses the pinned `Incident_simulator/satellite.tle` and
the author's historical example time. It is outside the location dataset's scope.
