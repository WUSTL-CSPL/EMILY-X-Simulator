#!/usr/bin/env python3
"""Check the complete 30-second preflight run before the million-sample run."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import generate_multireceiver_v8_30s_million as runner

g=runner.g
root=g.ROOT
request=g.load(root/'configs/multireceiver_v8_30s_million_smoke.json')
campaign=root/request['campaign_dir']
assert g.load(campaign/'COMPLETE.json')['all_requested_datasets_complete']
reference=g.load(campaign/'restart_reference.json')
db=sqlite3.connect('file:'+str(campaign/'scene_cache.sqlite')+'?mode=ro',uri=True)
old=sqlite3.connect('file:'+str(root/'runs/v8_duration_comparison/bundles.sqlite')+'?mode=ro',uri=True)
ids=[r[0] for r in db.execute("select id from proposals where split='train' order by ordinal")]
assert ids[:len(reference['initial_ids'])]==reference['initial_ids']
assert db.execute('select count(*)=count(distinct id) from proposals').fetchone()[0]
replays=0
for sid,payload in db.execute('select id,payload from scenes'):
    found=old.execute('select payload from bundles where id=?',(sid,)).fetchone()
    if found:
        expected=json.loads(zlib.decompress(found[0]))['rows']['native_30']
        expected['truth'].pop('observation_duration_requested_s')
        expected['truth'].pop('duration_experiment_mode')
        assert json.loads(zlib.decompress(payload))==expected,sid
        replays+=1
counts={};all_ids={};three_only=0
for n in (3,4):
    output=root/f"{request['output_prefix']}_{n}rx_20km_1000"
    assert g.load(output/'COMPLETE.json')['samples']==1000
    counts[str(n)]={};all_ids[n]={}
    config=g.load(output/'config.json')
    assert config['duration_s']==config['simulation']['duration_s']==30
    for split,needed in request['small_splits'].items():
        selected=[r[0] for r in db.execute('select id from selections where split=? and n=? order by rank',(split,n))]
        actual=[]
        for part in sorted((output/split).glob('part-*')):
            info=g.load(part/'manifest.json')
            runner.verify_part(part,info['count'])
            with gzip.open(part/'inputs.jsonl.gz','rt') as fi, gzip.open(part/'labels.jsonl.gz','rt') as fl, gzip.open(part/'ground_truth.jsonl.gz','rt') as ft:
                for il,ll,tl in zip(fi,fl,ft,strict=True):
                    inp,lab,truth=map(json.loads,(il,ll,tl))
                    sid=inp.pop('sample_id');assert sid==lab.pop('sample_id')==truth['sample_id']
                    row=dict(sample_id=sid,input=inp,label=lab,truth=truth)
                    cached=json.loads(zlib.decompress(db.execute('select payload from scenes where id=?',(sid,)).fetchone()[0]))
                    assert row==g.subset_receivers(cached,n)
                    runner.validate_duration(row,30);g.validate_randomization(row,n,request)
                    assert g.all_detected(row,n) and truth['split']==split
                    if n==3 and not g.all_detected(cached,4):three_only+=1
                    actual.append(sid)
        assert actual==selected[:needed]
        assert len(actual)==len(set(actual))==needed
        counts[str(n)][split]=len(actual);all_ids[n][split]=set(actual)
    assert not any(all_ids[n][a]&all_ids[n][b] for a,b in [('train','validation'),('train','test'),('validation','test')])
assert three_only>0,'3-RX must not be restricted to all-four detections'
protected=g.load(campaign/'protected_originals.json')
assert all(g.base.sha256(p)==sha for p,sha in protected.items())
report=dict(passed=True,exact_native_30_replays=replays,initial_committed_rows_preserved=len(reference['initial_ids']),duplicate_proposals=False,export_counts=counts,three_rx_samples_without_rx4_detection=three_only,protected_originals_unchanged=True)
g.base.write_json(campaign/'preflight_verification.json',report)
print(json.dumps(report,indent=2))
