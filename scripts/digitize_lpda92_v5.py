#!/usr/bin/env python3
"""Extract numeric curves from the pinned manufacturer's PDF, not raw measurements.

Requires Poppler pdfimages, numpy and Pillow. No existing output is overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
from PIL import Image

URL = "https://drive.google.com/uc?export=download&id=1pup5J92OWVpsRE8j_v-te_hXMwyzA138"
PDF_SHA256 = "e3e25372315ba0dfb67355d5c230574cdc432121704f798a80964512c9a37c17"
# Image numbering is for pdfimages -f 4 -l 5 on this exact PDF.
# Centers/radii are calibrated from the outer 0 dB and inner -10/-20 dB rings.
PANELS = {
    "horizontal_low": (135, (228., 216.), 162., [(690, "blue"), (780, "green"), (870, "red")]),
    "vertical_low": (137, (315., 298.), 224., [(690, "blue"), (780, "green"), (870, "red")]),
    "horizontal_mid": (131, (315., 298.), 224., [(1710, "blue"), (1850, "green"), (1950, "red")]),
    "vertical_mid": (133, (315., 298.), 224., [(1710, "blue"), (1850, "green"), (1950, "red")]),
    "horizontal_high": (136, (315., 298.), 224., [(2170, "blue"), (2300, "green"), (2400, "red"), (2500, "yellow")]),
    "vertical_high": (138, (315., 298.), 224., [(2170, "blue"), (2300, "green"), (2400, "red"), (2500, "yellow")]),
}


def color_mask(rgb, name):
    r, g, b = np.moveaxis(rgb.astype(float), -1, 0)
    return {
        "red": (r > 140) & (g < 100) & (b < 100),
        "green": (g > 65) & (r < 95) & (b < 100) & (g > r + 35) & (g > b + 35),
        "blue": (b > 130) & (r < 100) & (g < 110),
        "yellow": (r > 100) & (g > 100) & (b < 100) & (abs(r-g) < 65),
    }[name]


def digitize_polar(path, center, radius, color):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    y, x = np.nonzero(color_mask(rgb, color))
    dx, dy = x - center[0], center[1] - y
    r = np.hypot(dx, dy)
    keep = (r <= radius + 2) & (r > 2)
    r, angle = r[keep], np.degrees(np.arctan2(dy[keep], dx[keep])) % 360
    values = np.full(360, np.nan)
    for a in range(360):
        near = np.abs((angle-a+180) % 360-180) <= 0.75
        if np.any(near):
            values[a] = np.clip(50 * (np.median(r[near])/radius - 1), -50, 0)
    # Near boresight the colored curves overlap heavily. For an occluded curve
    # use the visible RGB main-lobe envelope from the SAME panel. Yellow/cyan/
    # magenta are excluded because the dashed -3 dB reference ring uses them.
    # This is a documented reconstruction, not recovered raw measurement data.
    union = color_mask(rgb, "red") | color_mask(rgb, "green") | color_mask(rgb, "blue")
    uy, ux = np.nonzero(union)
    ur = np.hypot(ux-center[0], center[1]-uy)
    ua = np.degrees(np.arctan2(center[1]-uy, ux-center[0])) % 360
    reconstructed = 0
    for a in range(360):
        if min(a, 360-a) > 45 or np.isfinite(values[a]):
            continue
        near = (np.abs((ua-a+180) % 360-180) <= 0.75) & (ur <= radius+2) & (ur > 2)
        if np.any(near):
            values[a] = np.clip(50*(np.percentile(ur[near], 90)/radius-1), -50, 0)
            reconstructed += 1
    # Overlapping colored curves can hide each other. Interpolate these gaps.
    # All manufacturer polar cuts are normalized to the forward lobe at 0 deg.
    values[0] = 0.
    good = np.flatnonzero(np.isfinite(values))
    gaps = np.diff(np.r_[good, good[0]+360])
    if len(good) < 100 or np.max(gaps) > 100:
        raise ValueError(f"Insufficient visible curve: {path.name}/{color}")
    out = np.interp(np.arange(360), np.r_[good-360, good, good+360],
                    np.tile(values[good], 3))
    # Use a 5-degree table; fine structure below plot resolution is not claimed.
    return out[::5].round(4).tolist(), {"visible_angle_bins": len(good),
        "main_lobe_envelope_reconstructed_bins": reconstructed,
        "largest_interpolated_gap_deg": int(np.max(gaps)),
        "pixel_scale_db": 50/radius}


def digitize_gain(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    # Chart axes: x=83 -> 600 MHz, x=570 -> 3900 MHz;
    # y=35 -> 16 dBi, y=405 -> -5 dBi. Last tick before right edge is 3800.
    mask = color_mask(rgb, "blue")
    xs, gs = [], []
    for x in range(84, 570):
        ys = np.flatnonzero(mask[35:406, x]) + 35
        if len(ys):
            xs.append(600 + (x-83)/487*3300)
            gs.append(16 - (np.median(ys)-35)/370*21)
    freqs = np.arange(690., 2500.1, 10.)
    return freqs.tolist(), np.interp(freqs, xs, gs).round(4).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Use an already downloaded manufacturer PDF")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.TemporaryDirectory(prefix="lpda92-") as tmp:
        tmp = Path(tmp)
        pdf = args.pdf
        if pdf is None:
            pdf = tmp / "manufacturer.pdf"
            urllib.request.urlretrieve(URL, pdf)
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
            raise ValueError("Manufacturer PDF version differs; recalibrate extraction before using it")
        subprocess.run(["pdfimages", "-f", "4", "-l", "5", "-png", str(pdf), str(tmp/"img")], check=True)
        records = {"horizontal": {}, "vertical": {}}
        audit = {}
        for panel, (index, center, radius, curves) in PANELS.items():
            plane = panel.split("_")[0]
            for freq, color in curves:
                curve, check = digitize_polar(tmp/f"img-{index:03d}.png", center, radius, color)
                records[plane][freq] = curve
                audit[f"{plane}_{freq}_MHz"] = check
        freq, gain = digitize_gain(tmp/"img-037.png")
        result = {"model": "POYNTING LPDA-92", "source": {
            "product_url": "https://poynting.tech/antennas/lpda-92/",
            "pdf_url": URL, "pdf_sha256": PDF_SHA256,
            "document": "TS-A-LPDA-0092-V1 REV 7.0, February 2023",
            "pages": {"gain": 4, "patterns": 5},
            "data_kind": "Digitized manufacturer PDF curves; not original chamber measurement tables",
            "extraction_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "angles_deg_ccw": list(range(0, 360, 5)),
            "pattern_frequencies_mhz": sorted(records["horizontal"]),
            "horizontal_relative_db": [records["horizontal"][f] for f in sorted(records["horizontal"])],
            "vertical_relative_db": [records["vertical"][f] for f in sorted(records["vertical"])],
            "gain_frequencies_mhz": freq, "peak_gain_dbi": gain,
            "digitization_audit": audit,
            "limits": ["Angles interpolate linearly in dB; overlapping plot curves require gap interpolation.",
                       "Occluded main-lobe angles within +/-45 deg use the same-panel visible RGB envelope.",
                       "Patterns interpolate linearly in dB between published frequencies, including the 870-1710 MHz gap.",
                       "The 1600 MHz harmonic pattern is interpolated, not measured at 1600 MHz.",
                       "Horizontal and vertical cuts do not determine a unique full 3D pattern.",
                       "No measured accuracy claim is made for the digitized curves."]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
