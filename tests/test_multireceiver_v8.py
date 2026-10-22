"""Multi-site sampling, conditioning and observation-boundary tests."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_multireceiver_dataset_v8 as g

class MultiReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=g.load(g.ROOT/'configs/multireceiver_v8.json')
        cls.config=g.load(g.ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
        cls.receivers=g.receivers_for(cls.request,cls.config)

    def test_all_receiver_condition_is_strict(self):
        for missing in range(4):
            groups=[{'receiver_id':f'RX{i+1}','incidents':[] if i==missing else [{}]} for i in range(4)]
            row={'input':{'receivers':groups}}
            self.assertFalse(g.all_detected(row,4))
            self.assertEqual(g.all_detected(row,3),missing==3)

    def test_trimming_cannot_leak_fourth_receiver_or_modify_original(self):
        row={'input':{'receivers':[{'receiver_id':f'RX{i+1}','incidents':[{'value':i}]} for i in range(4)]},
             'truth':{'receivers':[{'name':f'RX{i+1}'} for i in range(4)],'receiver_seeds':[1,2,3,4],
                      'links':{f'RX{i+1}':{'value':i} for i in range(4)}}}
        before=deepcopy(row);trimmed=g.subset_receivers(row,3)
        self.assertEqual(row,before)
        self.assertEqual(len(trimmed['input']['receivers']),3)
        self.assertNotIn('RX4',trimmed['truth']['links'])
        self.assertEqual(trimmed['truth']['receiver_seeds'],[1,2,3])

    def test_sites_preserve_original_hardware_and_noncollinear_geometry(self):
        self.assertEqual(self.receivers[:2],list(g.base.ORIGINAL_RXS))
        forward,_=g.base.projection(self.config['profile'])
        xy=g.np.array([forward.transform(r.lon_deg,r.lat_deg) for r in self.receivers])
        for r in self.receivers:
            self.assertEqual((r.antenna_gain_dbi,r.height_agl_m,r.noise_figure_db),(0.,2.,6.))
        a,b,c=xy[:3];area=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
        self.assertGreater(area,50e6)

    def test_proposals_reproduce_and_keep_geographic_roles_across_domains(self):
        stored={};owners={}
        roles=g.load(g.ROOT/'datasets/terrestrial_location_v3/split_manifest.json')['block_roles']
        for width in (20,40):
            blocks={b['block_id']:b for b in g.blocks_for(width,roles,self.request['extension_split_seed'])}
            for split in g.SPLITS:
                a=g.Proposals(width,split,self.request,self.config,self.receivers)
                b=g.Proposals(width,split,self.request,self.config,self.receivers)
                seen=set()
                for _ in range(300):
                    x=a.next();self.assertEqual(x,b.next());self.assertNotIn(x['sample_id'],seen);seen.add(x['sample_id'])
                    self.assertEqual(owners.setdefault(x['block_id'],split),split)
                    if x['sample_id'] in stored:self.assertEqual(x,stored[x['sample_id']])
                    stored[x['sample_id']]=x
                    self.assertEqual(blocks[x['block_id']]['split'],split)
                    self.assertLess(abs(x['x_m']),width*500)
                    self.assertLess(abs(x['y_m']),width*500)
                    distances=[g.base.sim.GEOD.inv(x['longitude_deg'],x['latitude_deg'],r.lon_deg,r.lat_deg)[2] for r in self.receivers]
                    self.assertGreaterEqual(min(distances),1000.)

if __name__=='__main__':unittest.main()
