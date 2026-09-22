"""Checks for the merged CLI, custom regions and safe resumption."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_location_dataset as gen


class UnifiedGeneratorTests(unittest.TestCase):
    def test_original_and_compatibility_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["--stage", "train", "--output", str(Path(tmp)/"out")]
            original = gen.parse_args(argv)
            compat = gen.parse_args(argv, default_profile="near_south10", default_count=2000,
                                    default_incident_target=1000, default_workers=8)
            self.assertEqual((original.profile, original.record_format, original.count,
                              original.min_distance_km), ("original", "v1", 1000, 0.))
            self.assertEqual((compat.profile, compat.record_format, compat.count,
                              compat.min_distance_km, compat.incident_target),
                             ("near_south10", "v2", 2000, 1., 1000))

    def test_custom_region_keeps_strata_and_coordinates(self):
        profile = {**gen.PROFILES["original"], "region": "custom",
                   "latitude_deg": 39.47, "longitude_deg": -114.45}
        points = gen.sample_positions("train", 100, profile, 9, width_km=10., cells=5)
        forward, _ = gen.projection(profile)
        counts = {}
        for row in points:
            cell = row["cell_x"], row["cell_y"]
            counts[cell] = counts.get(cell, 0) + 1
            x, y = forward.transform(row["longitude_deg"], row["latitude_deg"])
            self.assertAlmostEqual(x, row["x_m"], places=5)
            self.assertAlmostEqual(y, row["y_m"], places=5)
            self.assertTrue(-5000 <= x <= 5000 and -5000 <= y <= 5000)
            for rx in gen.ORIGINAL_RXS:
                self.assertGreaterEqual(gen.sim.GEOD.inv(row["longitude_deg"], row["latitude_deg"],
                                                       rx.lon_deg, rx.lat_deg)[2], 1000.)
        self.assertEqual(len(counts), 25)
        self.assertEqual(set(counts.values()), {4})

    def test_resume_round_trip_and_changed_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/"out"
            config = {"start_utc": gen.sim.SIM_START_UTC, "srtm_dir": tmp, "seed": 1}
            scenes = [{"sample_id": "example"}]
            # A saved JSON manifest must compare equal after reading it back.
            manifest = gen.runtime_manifest(config)
            self.assertEqual(manifest, json.loads(json.dumps(manifest)))
            gen.prepare_output(out, config, scenes)
            gen.prepare_output(out, config, scenes, resume=True)
            with self.assertRaisesRegex(ValueError, "configuration"):
                gen.prepare_output(out, {**config, "seed": 2}, scenes, resume=True)
            with self.assertRaisesRegex(ValueError, "scenario list"):
                gen.prepare_output(out, config, [], resume=True)
            saved = json.loads((out/"manifest.json").read_text())
            saved["terrain_sha256"]["changed.hgt"] = "different"
            gen.write_json(out/"manifest.json", saved)
            with self.assertRaisesRegex(ValueError, "environment or terrain"):
                gen.prepare_output(out, config, scenes, resume=True)

    def test_cli_rejects_incomplete_custom_center_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                gen.parse_args(["--stage", "train", "--profile", "custom",
                                "--center-lat", "39.47", "--output", str(Path(tmp)/"out")])
            with self.assertRaises(SystemExit):
                gen.parse_args(["--stage", "train", "--output", tmp])


if __name__ == "__main__":
    unittest.main()
