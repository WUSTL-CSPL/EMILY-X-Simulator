"""Geographic leakage and reproducibility checks for the paired datasets."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_location_comparison_v6 as g

class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=g.load(g.ROOT/'configs/location_comparison_v6.json')
        cls.c=g.load(g.ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
        cls.roles=g.load(g.ROOT/'datasets/terrestrial_location_v3/split_manifest.json')['block_roles']
        cls.data={w:g.sample_dataset(w,cls.r,cls.c,cls.roles) for w in (20,40)}

    def test_same_seed_reproduces_all_scenarios(self):
        self.assertEqual(self.data[20],g.sample_dataset(20,self.r,self.c,self.roles))

    def test_valid_separation_and_exclusion_in_both_domains(self):
        for w,(scenes,blocks,quotas) in self.data.items():
            g.verify_geometry(scenes,blocks,self.r,self.c)
            self.assertEqual(sum(len(v) for v in scenes.values()),10000)

    def test_shared_scenes_cannot_cross_split_roles(self):
        owner={}; seen={}
        for w,(scenes,blocks,quotas) in self.data.items():
            for split,items in scenes.items():
                for s in items:
                    sid=s['sample_id']
                    if sid in seen:
                        self.assertEqual(s,seen[sid])
                        self.assertEqual(split,owner[sid])
                    seen[sid]=s; owner[sid]=split
        for split in ('validation','test'):
            a={s['sample_id'] for s in self.data[20][0][split]}
            b={s['sample_id'] for s in self.data[40][0][split]}
            self.assertGreater(len(a & b),0)
        # Independently check that no held-out geographic block has become training.
        by_block={}
        for _,blocks,_ in self.data.values():
            for b in blocks:
                self.assertEqual(by_block.setdefault(b['block_id'],b['split']),b['split'])

if __name__=='__main__': unittest.main()
