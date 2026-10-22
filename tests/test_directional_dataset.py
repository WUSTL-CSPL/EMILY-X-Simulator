"""Directional response, noise placement and empty-scene checks."""
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_directional_location_dataset as directional

np = directional.np
sim = directional.base.sim
PATTERN = {"model": "horizontal_parabolic", "pointing": "fixed_toward_baseline_region_center",
           "horizontal_hpbw_deg": 120.0, "peak_gain_dbi": 0.0,
           "max_attenuation_db": 20.0, "vertical_pattern": "none"}


class DirectionalTests(unittest.TestCase):
    def test_half_power_width_sides_back_and_angle_wrap(self):
        for angle, expected in [(0, 0), (60, -3), (-60, -3), (90, -6.75),
                                (180, -20), (360, 0), (420, -3)]:
            self.assertAlmostEqual(directional.gain_dbi(angle, PATTERN), expected)

    def test_fixed_pointing_uses_receiver_to_source_bearing(self):
        rx = directional.base.ORIGINAL_RXS[0]
        pattern = {**PATTERN, "azimuth_deg": 359.0}
        lon, lat, _ = sim.GEOD.fwd(rx.lon_deg, rx.lat_deg, 1.0, 10000)
        result = directional.evaluate_pattern(rx, {"longitude_deg": lon, "latitude_deg": lat}, pattern)
        self.assertAlmostEqual(result["off_axis_deg"], 2.0, places=8)
        self.assertEqual(pattern["azimuth_deg"], 359.0)

    def test_invalid_or_nonfinite_parameters_rejected(self):
        for key, value in [("horizontal_hpbw_deg", 0), ("horizontal_hpbw_deg", float('nan')),
                           ("max_attenuation_db", -1), ("peak_gain_dbi", 3)]:
            with self.assertRaises(ValueError):
                directional.validate_pattern({**PATTERN, key: value})

    def test_effective_gain_attenuates_source_but_not_noise(self):
        """Exercise upstream signal+noise mixing, rather than just the formula."""
        freq = np.array([800e6, 801e6])
        comp = SimpleNamespace(name="LTE", center_hz=800e6, kind="fundamental",
                               total_power_dbm=0.0, harmonic_order=1, antenna_extra_loss_db=0.0)
        models = {"loaded": {"templates": [{"template_index": 0, "components": [comp],
                   "fractions": {"LTE": np.array([0.5, 0.5])}}]}}
        prop = {"geometry": {"bearing_tx_to_rx_deg": 0., "elevation_tx_to_rx_deg": 0.},
                "mode": "test", "terrain": None, "path_loss_db": np.array([80., 80.]),
                "center_losses_db": {str(int(800e6)): 80.}}
        def run(gain, state):
            rx = replace(directional.base.ORIGINAL_RXS[0], antenna_gain_dbi=gain)
            return sim.make_receiver_spectrogram_v3(rx, directional.base.ORIGINAL_TX,
                freq, models, np.full(60, state), np.zeros(60, dtype=int), prop,
                60, np.random.default_rng(123))[0]
        with patch.object(sim, "tx_antenna_gain_db", return_value=0.0):
            noise = run(0., "off")
            np.testing.assert_array_equal(noise, run(-20., "off"))
            original = run(0., "loaded")
            attenuated = run(-20., "loaded")
        signal_a = 10. ** (original.astype(float) / 10.) - 10. ** (noise.astype(float) / 10.)
        signal_b = 10. ** (attenuated.astype(float) / 10.) - 10. ** (noise.astype(float) / 10.)
        np.testing.assert_allclose(signal_b / signal_a, 0.01, rtol=1e-4)

    def test_empty_scene_is_counted_not_dropped(self):
        empty = {"receivers": [{"incidents": []}, {"incidents": []}]}
        detected = copy.deepcopy(empty)
        detected["receivers"][0]["incidents"].append({"example": True})
        counts = directional.coverage([empty, detected])
        self.assertEqual(counts["samples"], 2)
        self.assertEqual(counts["neither"], 1)
        self.assertEqual(counts["any_detection_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
