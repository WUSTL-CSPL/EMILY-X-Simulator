"""Bounded-memory runner: deterministic checkpoints and real uncached simulation."""
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_multireceiver_v8_large as large

g=large.g

class LargeRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request=g.load(g.ROOT/'configs/multireceiver_v8_large_smoke.json')
        cls.config=g.load(g.ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
        cls.receivers=g.receivers_for(cls.request,cls.config)

    def test_rng_checkpoint_reproduces_next_proposals(self):
        stream=g.Proposals(20,'train',self.request,self.config,self.receivers)
        for _ in range(137):stream.next()
        state=large.initial_state();state['rng']=stream.rng.bit_generator.state;state['counters']=dict(stream.counters)
        state=json.loads(json.dumps(state))
        resumed=large.stream_for(20,'train',self.request,self.config,self.receivers,state)
        for _ in range(100):self.assertEqual(stream.next(),resumed.next())

    def test_uncached_rf_matches_old_v8_and_resume_does_not_duplicate(self):
        request={**self.request,'batch_size':4}
        freq,models=g.base.build_models()
        g.initialize_worker(freq,models,self.config,self.receivers)
        with tempfile.TemporaryDirectory() as tmp:
            db=large.connect(Path(tmp)/'cache.sqlite')
            # One worker thread avoids shared mutable simulator state; production uses processes.
            with ThreadPoolExecutor(max_workers=1) as pool:
                result=large.advance(db,None,pool,20,'train',14,request,self.config,self.receivers,Path(tmp),[1])
                self.assertFalse(result)
                state=large.get_state(db,20,'train')
                self.assertEqual(state['proposed'],4);self.assertEqual(state['new_simulations'],4)
                first_ids=[r[0] for r in db.execute('select id from proposals order by ordinal')]
                large.advance(db,None,pool,20,'train',14,request,self.config,self.receivers,Path(tmp),[1])
                self.assertEqual(large.get_state(db,20,'train')['proposed'],8)
                ids=[r[0] for r in db.execute('select id from proposals order by ordinal')]
                self.assertEqual(ids[:4],first_ids);self.assertEqual(len(set(ids)),8)
            with sqlite3.connect('file:'+str(g.ROOT/'runs/multireceiver_v8/scene_cache.sqlite')+'?mode=ro',uri=True) as old:
                for sid,payload in db.execute('select id,payload from scenes'):
                    expected=old.execute('select payload from scenes where id=?',(sid,)).fetchone()[0]
                    self.assertEqual(json.loads(large.zlib.decompress(payload)),json.loads(large.zlib.decompress(expected)))
            db.close()

if __name__=='__main__':unittest.main()
