"""Checks for sampling, feature leakage and deterministic scene isolation."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import generate_location_dataset as gen


class SamplingTests(unittest.TestCase):
    def test_strata_seed_and_coordinate_round_trip(self):
        a = gen.sample_scenarios("train", 1000, 10, 20., 1234)
        self.assertEqual(a, gen.sample_scenarios("train", 1000, 10, 20., 1234))
        self.assertNotEqual(a, gen.sample_scenarios("train", 1000, 10, 20., 1235))
        self.assertEqual(len({(p["latitude_deg"], p["longitude_deg"]) for p in a}), 1000)
        cells = np.zeros((10, 10), dtype=int)
        forward, _ = gen.projections()
        for p in a:
            cells[p["cell_y"], p["cell_x"]] += 1
            x, y = forward.transform(p["longitude_deg"], p["latitude_deg"])
            self.assertLess(abs(x - p["x_m"]), 1e-5)
            self.assertLess(abs(y - p["y_m"]), 1e-5)
            self.assertTrue(-10000 <= x <= 10000 and -10000 <= y <= 10000)
        np.testing.assert_array_equal(cells, np.full((10, 10), 10))

    def test_pilot_and_train_have_independent_noise_streams(self):
        a = gen.sample_scenarios("pilot", 100, 10, 20., 1234)
        b = gen.sample_scenarios("train", 100, 10, 20., 1234)
        self.assertTrue(set(s for p in a for s in p["receiver_seeds"]).isdisjoint(
            s for p in b for s in p["receiver_seeds"]))

    def test_invalid_stratification_rejected(self):
        with self.assertRaises(ValueError):
            gen.sample_scenarios("train", 999, 10, 20., 1)


class OutputTests(unittest.TestCase):
    def setUp(self):
        folder = gen.ROOT / "datasets/terrestrial_location_v1/pilot/results"
        paths = sorted(folder.glob("*.json"))
        if not paths:
            folder = gen.ROOT / "datasets/terrestrial_location_v2/campaign/results"
            paths = sorted(folder.glob("*.json"))
        if not paths:
            self.skipTest("Generate a pilot or the v3 source pool for output-boundary tests")
        rows = (json.loads(p.read_text()) for p in paths)
        self.row = next(r for r in rows if any(g["incidents"] for g in r["input"]["receivers"]))

    def test_current_output_valid(self):
        gen.validate_row(self.row)

    def test_transmitter_truth_in_input_rejected(self):
        self.row["input"]["receivers"][0]["incidents"][0]["true_transmitter_lat"] = 39.2
        with self.assertRaises(AssertionError):
            gen.validate_row(self.row)

    def test_fabricated_end_of_ongoing_incident_rejected(self):
        inc = self.row["input"]["receivers"][0]["incidents"][0]
        self.assertTrue(inc["ongoing"])
        inc["t_end_offset_sec"] = 59
        with self.assertRaises(AssertionError):
            gen.validate_row(self.row)


if __name__ == "__main__":
    unittest.main()
