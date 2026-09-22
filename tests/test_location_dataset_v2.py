"""Coverage-campaign geometry, isolation and observation-boundary checks."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_location_dataset as v2
import numpy as np


class SamplingTests(unittest.TestCase):
    def test_original_profile_keeps_v1_positions_and_noise(self):
        old = v2.sample_scenarios("pilot", 100, 10, 20., 20260921)
        new = v2.sample_positions("pilot", 100, v2.PROFILES["original"], 20260921)
        for a, b in zip(old, new):
            self.assertAlmostEqual(a["latitude_deg"], b["latitude_deg"], places=12)
            self.assertAlmostEqual(a["longitude_deg"], b["longitude_deg"], places=12)
            self.assertEqual(a["receiver_seeds"], b["receiver_seeds"])

    def test_new_regions_keep_counts_and_minimum_link_distance(self):
        for name in ("near_midpoint", "near_south10", "near_south20"):
            profile = v2.PROFILES[name]
            rows = v2.sample_positions("train", 1000, profile, 20260921)
            self.assertEqual(rows, v2.sample_positions("train", 1000, profile, 20260921))
            self.assertEqual(len({(r["latitude_deg"], r["longitude_deg"]) for r in rows}), 1000)
            counts = np.zeros((10, 10), dtype=int)
            fwd, _ = v2.projection(profile)
            for row in rows:
                x, y = fwd.transform(row["longitude_deg"], row["latitude_deg"])
                self.assertAlmostEqual(x, row["x_m"], places=5)
                self.assertAlmostEqual(y, row["y_m"], places=5)
                self.assertTrue(-10000 <= x <= 10000 and -10000 <= y <= 10000)
                counts[row["cell_y"], row["cell_x"]] += 1
                for rx in v2.ORIGINAL_RXS:
                    d = v2.sim.GEOD.inv(row["longitude_deg"], row["latitude_deg"], rx.lon_deg, rx.lat_deg)[2]
                    self.assertGreaterEqual(d, 1000)
            np.testing.assert_equal(counts, np.full((10, 10), 10))

    def test_sampler_does_not_mutate_receivers(self):
        old = [(r.lon_deg, r.lat_deg, r.height_agl_m) for r in v2.sim.RECEIVERS]
        v2.sample_positions("pilot", 100, v2.PROFILES["near_south10"], 1)
        self.assertEqual(old, [(r.lon_deg, r.lat_deg, r.height_agl_m) for r in v2.sim.RECEIVERS])


class RegressionTests(unittest.TestCase):
    def test_original_pilot_observations_equal_v1(self):
        root = v2.ROOT / "datasets"
        new_path = root / "terrestrial_location_v2/survey/original/inputs.jsonl"
        if not new_path.exists():
            self.skipTest("Survey must finish first")
        old = [json.loads(x) for x in (root / "terrestrial_location_v1/pilot/inputs.jsonl").read_text().splitlines()]
        new = [json.loads(x) for x in new_path.read_text().splitlines()]
        self.assertEqual(len(old), len(new))
        for a, b in zip(old, new):
            self.assertEqual(a["receivers"], b["receivers"])


if __name__ == "__main__":
    unittest.main()
