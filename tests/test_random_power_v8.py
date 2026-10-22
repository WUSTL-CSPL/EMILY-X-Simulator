"""Check that random power reaches RF output and offsets reach observations."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_multireceiver_dataset_v8 as g


class PerturbationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=g.load(g.ROOT/'configs/multireceiver_v8.json')
        cls.config=g.load(g.ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
        cls.receivers=g.receivers_for(cls.request,cls.config)
        cls.freq,cls.models=g.base.build_models()

    def test_physical_signal_scales_but_thermal_floor_does_not(self):
        original=self.models['loaded']['templates'][0]['components'][0].total_power_dbm
        scaled=g.scale_models(self.models,-7.)
        self.assertEqual(self.models['loaded']['templates'][0]['components'][0].total_power_dbm,original)
        self.assertIs(g.scale_models(self.models,0.),self.models)
        for a,b in zip(self.models['loaded']['templates'],scaled['loaded']['templates']):
            self.assertEqual([c.total_power_dbm-7 for c in a['components']],[c.total_power_dbm for c in b['components']])
            self.assertIs(a['fractions'],b['fractions'])
        prop=dict(geometry={'bearing_tx_to_rx_deg':10.,'elevation_tx_to_rx_deg':-4.},
                  path_loss_db=g.np.full(len(self.freq),130.),center_losses_db={},mode='test_constant_loss',terrain=None)
        def spectrum(models,state='loaded'):
            return g.ORIGINAL_SPECTROGRAM(self.receivers[0],g.base.ORIGINAL_TX,self.freq,models,
                g.np.full(4,state,dtype='U8'),g.np.arange(4),prop,4,g.np.random.default_rng(923))
        high,truth=spectrum(self.models);low,low_truth=spectrum(scaled);noise,_=spectrum(self.models,'off')
        expected=g.base.sim.dbm_to_mw(noise)+(g.base.sim.dbm_to_mw(high)-g.base.sim.dbm_to_mw(noise))*10**(-.7)
        g.np.testing.assert_allclose(low,g.base.sim.mw_to_dbm(expected),atol=3e-5,rtol=0)
        self.assertEqual(truth['noise_floor_per_rbw_dbm'],low_truth['noise_floor_per_rbw_dbm'])
        for a,b in zip(truth['component_truth'],low_truth['component_truth']):
            self.assertAlmostEqual(a['median_tx_total_power_dbm']-7,b['median_tx_total_power_dbm'],places=4)
            self.assertAlmostEqual(a['median_rx_total_component_dbm']-7,b['median_rx_total_component_dbm'],places=4)
        shifted,audit=g.add_measurement_offset(high,truth,2.,2.)
        g.np.testing.assert_array_equal(shifted,high+g.np.float32(2.))
        self.assertIs(audit['component_truth'],truth['component_truth'])
        self.assertEqual(audit['measured_noise_floor_per_rbw_dbm'],truth['noise_floor_per_rbw_dbm']+2.)
        zero,_=g.add_measurement_offset(high,truth,0.,0.)
        g.np.testing.assert_array_equal(zero,high)

    def test_power_and_offset_streams_are_independent_and_reproducible(self):
        props=g.Proposals(20,'train',self.request,self.config,self.receivers)
        scenes=[props.next() for _ in range(3000)]
        values=g.np.array([[s['transmitter_power_dbm'],*s['receiver_measurement_offsets_db'].values()] for s in scenes])
        self.assertTrue(g.np.all((values[:,0]>=33)&(values[:,0]<=43)))
        self.assertLess(abs(values[:,0].mean()-38),.15)
        self.assertTrue(g.np.all(abs(values[:,1:].mean(axis=0))<.12))
        self.assertTrue(g.np.all(abs(values[:,1:].std(axis=0)-2)<.12))
        correlation=g.np.corrcoef(values.T)-g.np.eye(5)
        self.assertLess(abs(correlation).max(),.07)
        changed={**self.request,'measurement_offset_sigma_db':0.}
        changed_scene=g.Proposals(20,'train',changed,self.config,self.receivers).next()
        self.assertEqual(changed_scene['transmitter_power_dbm'],scenes[0]['transmitter_power_dbm'])
        self.assertEqual(changed_scene['receiver_seeds'],scenes[0]['receiver_seeds'])
        self.assertEqual(changed_scene['receiver_measurement_offsets_db'],dict.fromkeys(['RX1','RX2','RX3','RX4'],0.))

    def test_zero_change_reproduces_original_and_wrapper_restores_state(self):
        old=self.config
        path=g.ROOT/'datasets/terrestrial_location_v3/train/ground_truth.jsonl'
        import json
        with path.open() as handle:truth=json.loads(handle.readline())
        keys=('sample_id','x_m','y_m','latitude_deg','longitude_deg','cell_x','cell_y','receiver_seeds','geometry_replacements')
        scene={k:truth[k] for k in keys};scene['receiver_seeds']+=[7103,7104]
        g.initialize_worker(self.freq,self.models,old,self.receivers)
        original=g.simulate(scene)
        altered={**scene,'transmitter_power_dbm':43.,'receiver_measurement_offsets_db':dict.fromkeys(['RX1','RX2','RX3','RX4'],0.),'measurement_offset_sigma_db':0.}
        row=g.simulate(altered)
        self.assertEqual(row['input'],original['input'])
        for k in ('transmitter_power_dbm','receiver_measurement_offsets_db','measurement_offset_sigma_db'):row['truth'].pop(k)
        for link in row['truth']['links'].values():
            for k in ('measurement_offset_db','measurement_offset_sigma_db','measurement_model','measured_noise_floor_per_rbw_dbm'):link.pop(k)
        self.assertEqual(row,original)
        self.assertIs(g.base.sim.make_receiver_spectrogram_v3,g.ORIGINAL_SPECTROGRAM)
        self.assertIs(g.base.WORKER['models'],self.models)
        offset_scene={**altered,'receiver_measurement_offsets_db':{'RX1':2.,'RX2':-2.,'RX3':1.,'RX4':-1.}}
        noisy=g.simulate(offset_scene)
        # At least the measured incident intensities change; no source power in features.
        self.assertNotEqual(noisy['input'],original['input'])
        trim=g.subset_receivers(noisy,3)
        self.assertNotIn('RX4',trim['truth']['receiver_measurement_offsets_db'])
        self.assertEqual(trim['input']['receivers'],noisy['input']['receivers'][:3])
        self.assertEqual(g.simulate(scene),original)


if __name__=='__main__':unittest.main()
