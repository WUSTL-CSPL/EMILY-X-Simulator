#!/usr/bin/env python3
"""Verify a portable campaign's complete gzip datasets and aligned IDs."""
import argparse,gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/portable_generate_million.json');args=parser.parse_args()
    c=json.loads(args.config.read_text());campaign=ROOT/c['campaign_dir'];complete=json.loads((campaign/'COMPLETE.json').read_text())
    assert complete['all_requested_datasets_complete'] is True
    reference=json.loads((ROOT/'reference/v8/reference_cases.json').read_text());results={};replayed=0
    for n in (3,4):
        out=ROOT/f"{c['output_prefix']}_{n}rx_20km_{c['small_size']}";marker=json.loads((out/'COMPLETE.json').read_text())
        assert marker['samples']==c['small_size'] and marker['sizes']==c['small_splits'] and marker['shards_verified'] and marker['all_receivers_detected']
        assert marker['signature']==complete['signature'];index=json.loads((out/'shards.json').read_text());results[str(n)]={}
        for split,target in c['small_splits'].items():
            count=0;seen=set()
            for item in index[split]:
                part=out/item['path']
                assert part.resolve().parent==(out/split).resolve()
                assert item['first_rank']==count+1
                for name,want in item['sha256'].items():assert digest(part/name)==want
                handles=[gzip.open(part/(name+'.jsonl.gz'),'rt') for name in ('inputs','labels','ground_truth')]
                local=0
                try:
                    for rows in zip(*handles,strict=True):
                        inp,lab,truth=map(json.loads,rows);sid=inp['sample_id'];assert sid==lab['sample_id']==truth['sample_id'] and sid not in seen
                        seen.add(sid);assert len(inp['receivers'])==n and all(r['incidents'] for r in inp['receivers'])
                        assert truth['split']==split
                        if sid in reference and str(n) in reference[sid]:
                            row=dict(sample_id=sid,input={k:v for k,v in inp.items() if k!='sample_id'},label={k:v for k,v in lab.items() if k!='sample_id'},truth=truth)
                            actual=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
                            assert actual==reference[sid][str(n)],'Reference replay differs; inspect dependency/CPU differences'
                            replayed+=1
                        count+=1;local+=1
                finally:
                    for handle in handles:handle.close()
                assert local==item['count'] and item['last_rank']==count
            assert count==target;results[str(n)][split]=count
    print(json.dumps(dict(passed=True,counts=results,reference_replays=replayed),indent=2))

if __name__=='__main__':main()
