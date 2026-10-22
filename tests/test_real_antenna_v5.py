"""Checks for frequency-dependent receive gain and the copied spectrum function."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import receiver_antenna_v5 as model
import generate_real_antenna_dataset_v5 as generator

np = model.np
sim = model.sim
TABLE = Path(__file__).resolve().parents[1]/'configs/antennas/lpda92_digitized_v5.json'


class RealAntennaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.antenna = model.AntennaPattern(TABLE)

    def test_boresight_is_published_gain_and_wrap_is_continuous(self):
        self.assertAlmostEqual(float(self.antenna.gain(800e6, 0, 0)), 10.8351, places=4)
        for f in [800e6, 1600e6, 2400e6]:
            self.assertAlmostEqual(float(self.antenna.gain(f, 360, 0)), float(self.antenna.gain(f, 0, 0)))
            self.assertAlmostEqual(float(self.antenna.gain(f, -180, 0)), float(self.antenna.gain(f, 180, 0)))
        self.assertLess(float(self.antenna.gain(800e6, 180, 0)), 0.)

    def test_no_silent_frequency_extrapolation(self):
        for f in [200e6, 2600e6, float('nan')]:
            with self.assertRaises(ValueError): self.antenna.gain(f, 0., 0.)

    def test_principal_cuts_and_clockwise_convention(self):
        peak = float(self.antenna.gain(780e6, 0, 0))
        i = self.antenna.data['pattern_frequencies_mhz'].index(780)
        h = self.antenna.horizontal[i, 270//5]
        v = self.antenna.vertical[i, 10//5]
        self.assertAlmostEqual(float(self.antenna.gain(780e6, 90, 0))-peak, h)
        self.assertAlmostEqual(float(self.antenna.gain(780e6, 0, 10))-peak, v)

    def test_fixed_orientation_uses_center_not_sample_location(self):
        config = {'center_longitude_deg': -114.45433040440209, 'center_latitude_deg':39.47016783035665,
                  'receivers':[{'name': 'RX1','lon_deg':-114.373325, 'lat_deg':39.5249}]}
        pattern = {'model':'POYNTING LPDA-92','data_file':str(TABLE),
                   'pointing':'fixed_toward_baseline_region_center','elevation_deg':0.,
                   'additional_cable_loss_db':0.,'polarization':'matched_scalar_reference'}
        result = generator.resolve_patterns(config, pattern)
        self.assertAlmostEqual(result['RX1']['azimuth_deg'],228.93398141031122)

    def test_upstream_copy_identity_and_frequency_gain_before_noise(self):
        freq = np.array([800e6, 801e6])
        comp = SimpleNamespace(name='LTE', center_hz=800e6, kind='fundamental',
                               total_power_dbm=0., harmonic_order=1, antenna_extra_loss_db=0.)
        models = {'loaded': {'templates':[{'template_index':0,'components':[comp],
                                           'fractions':{'LTE':np.array([0.5,0.5])}}]}}
        propagation = {'geometry':{'bearing_tx_to_rx_deg':0.,'elevation_tx_to_rx_deg':0.},
            'mode':'test','terrain':None,'path_loss_db':np.array([80.,80.]),'center_losses_db':{'800000000':80.}}
        rx = replace(model.base.ORIGINAL_RXS[0],antenna_gain_dbi=0.)
        def run(fn, state, **kwargs):
            return fn(rx, model.base.ORIGINAL_TX, freq, models, np.full(60,state),
                      np.zeros(60,dtype=int), propagation,60,np.random.default_rng(456),**kwargs)
        gain = lambda f: np.interp(f,freq,[10.,-10.])
        with patch.object(sim,'tx_antenna_gain_db',return_value=0.):
            old, old_truth = run(sim.make_receiver_spectrogram_v3,'loaded')
            copied, copied_truth = run(model.make_receiver_spectrogram_v3,'loaded')
            np.testing.assert_array_equal(old,copied)
            self.assertEqual(sim.sanitize_json_value(old_truth),sim.sanitize_json_value(copied_truth))
            noise,_ = run(model.make_receiver_spectrogram_v3,'off')
            directed_noise,_ = run(model.make_receiver_spectrogram_v3,'off',gain_function=gain)
            np.testing.assert_array_equal(noise,directed_noise)
            directed,truth = run(model.make_receiver_spectrogram_v3,'loaded',gain_function=gain)
        mw = lambda x: 10**(x.astype(float)/10.)
        ratio = (mw(directed)-mw(noise))/(mw(old)-mw(noise))
        np.testing.assert_allclose(ratio[:,0],10.,rtol=1e-4)
        np.testing.assert_allclose(ratio[:,1],0.1,rtol=1e-4)
        self.assertAlmostEqual(truth['component_truth'][0]['median_rx_total_component_dbm']-
                               old_truth['component_truth'][0]['median_rx_total_component_dbm'],10.,places=4)


if __name__ == '__main__':
    unittest.main()
