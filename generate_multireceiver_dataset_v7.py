#!/usr/bin/env python3
"""Original omnidirectional RF chain with three/four fixed receiving sites."""
from __future__ import annotations
import argparse
from collections import Counter
import concurrent.futures as futures
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sqlite3
import time

import generate_location_dataset as base
np=base.np
ROOT=Path(__file__).resolve().parent
SPLITS=('train','validation','test')
TABLES=('inputs.jsonl','labels.jsonl','ground_truth.jsonl')


def load(path): return json.loads(Path(path).read_text())


def receivers_for(request, config):
    _,inverse=base.projection(config['profile'])
    receivers=list(base.ORIGINAL_RXS)
    for site in request['added_receivers']:
        lon,lat=inverse.transform(site['east_m'],site['north_m'])
        receivers.append(replace(receivers[0],name=site['name'],lat_deg=lat,lon_deg=lon))
    assert [r.name for r in receivers]==['RX1','RX2','RX3','RX4']
    assert all(r.antenna_gain_dbi==0 and r.height_agl_m==2 for r in receivers)
    return receivers


def blocks_for(width, original_roles, seed):
    half=width*500.; blocks=[]
    for iy in range(math.floor((-half+10000)/4000),math.ceil((half+10000)/4000)):
        for ix in range(math.floor((-half+10000)/4000),math.ceil((half+10000)/4000)):
            core=0<=ix<5 and 0<=iy<5
            blocks.append(dict(block_id=f'{ix}_{iy}',cell_x=ix,cell_y=iy,core=core,
                bounds_m=[max(-half,-10000+ix*4000),min(half,-10000+(ix+1)*4000),
                          max(-half,-10000+iy*4000),min(half,-10000+(iy+1)*4000)],
                split=original_roles[str(iy*5+ix)] if core else None))
    outer=[b for b in blocks if not b['core']]
    order=np.random.default_rng(seed).permutation(len(outer)); n=round(.15*len(outer))
    for j,i in enumerate(order): outer[i]['split']='test' if j<n else 'validation' if j<2*n else 'train'
    return blocks


def initialize_worker(freq,models,config,receivers):
    base.initialize_worker(freq,models,config)
    base.sim.RECEIVERS=[replace(r) for r in receivers]


def simulate(scenario):
    row=base.simulate_scene(scenario)
    base.validate_row(row)
    return row


def all_detected(row,n):
    return len(row['input']['receivers'])>=n and all(r['incidents'] for r in row['input']['receivers'][:n])


def subset_receivers(row,n):
    result=deepcopy(row)
    result['input']['receivers']=result['input']['receivers'][:n]
    names=[r['receiver_id'] for r in result['input']['receivers']]
    result['truth']['receivers']=result['truth']['receivers'][:n]
    result['truth']['receiver_seeds']=result['truth']['receiver_seeds'][:n]
    result['truth']['links']={k:v for k,v in result['truth']['links'].items() if k in names}
    return result


class Proposals:
    def __init__(self,width,split,request,config,receivers,seed=None):
        self.seed=request['sampling_seed'] if seed is None else seed
        self.request,self.receivers=request,receivers
        self.width,self.split=width,split
        roles=load(ROOT/'datasets/terrestrial_location_v3/split_manifest.json')['block_roles']
        self.blocks=[b for b in blocks_for(width,roles,request['extension_split_seed']) if b['split']==split]
        buffer=request['boundary_buffer_m']
        areas=np.array([(b['bounds_m'][1]-b['bounds_m'][0]-2*buffer)*
                        (b['bounds_m'][3]-b['bounds_m'][2]-2*buffer) for b in self.blocks])
        self.probability=areas/areas.sum()
        self.rng=np.random.default_rng(np.random.SeedSequence([self.seed,701,width,SPLITS.index(split)]))
        _,self.inverse=base.projection(config['profile'])
        self.counters=Counter()

    def next(self):
        b=self.blocks[int(self.rng.choice(len(self.blocks),p=self.probability))]
        ix,iy=b['cell_x'],b['cell_y']; k=self.counters[b['block_id']];self.counters[b['block_id']]+=1
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,702,ix+100,iy+100,k]))
        buffer=self.request['boundary_buffer_m'];xmin,xmax,ymin,ymax=b['bounds_m']
        for retries in range(10000):
            x=float(rng.uniform(xmin+buffer,xmax-buffer));y=float(rng.uniform(ymin+buffer,ymax-buffer))
            lon,lat=self.inverse.transform(x,y)
            d=[base.sim.GEOD.inv(lon,lat,r.lon_deg,r.lat_deg)[2] for r in self.receivers]
            if min(d)>=self.request['min_distance_to_receiver_m']: break
        else: raise RuntimeError('No admissible geometry in block')
        sid='v7_'+hashlib.sha256(f'{self.seed}:{ix}:{iy}:{k}'.encode()).hexdigest()[:24]
        return dict(sample_id=sid,x_m=x,y_m=y,latitude_deg=lat,longitude_deg=lon,
            cell_x=ix,cell_y=iy,block_id=b['block_id'],split=self.split,
            region='core' if b['core'] else 'outer',proposal_index_in_block=k,
            geometry_replacements=retries,receiver_seeds=[int(s) for s in np.random.SeedSequence(
                [self.seed,703,ix+100,iy+100,k]).generate_state(4)])


def pilot(request):
    out=ROOT/'runs/multireceiver_v7_pilot'
    if out.exists(): raise FileExistsError(out)
    c=load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    receivers=receivers_for(request,c);freq,models=base.build_models()
    rows={}; started=time.monotonic()
    with futures.ProcessPoolExecutor(max_workers=request['workers'],mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker,initargs=(freq,models,c,receivers)) as pool:
        for width in (20,40):
            stream=Proposals(width,'train',request,c,receivers,seed=request['pilot_seed'])
            scenarios=[stream.next() for _ in range(request['pilot_per_width'])]
            rows[width]=list(pool.map(simulate,scenarios))
    summary={w:dict(proposals=len(r),all3=sum(all_detected(x,3) for x in r),
        all4=sum(all_detected(x,4) for x in r),per_receiver={f'RX{i+1}':sum(bool(x['input']['receivers'][i]['incidents']) for x in r) for i in range(4)},
        receiver_ground_elevation_m={name:float(np.median([x['truth']['links'][name]['rx_ground_elevation_m'] for x in r])) for name in ['RX1','RX2','RX3','RX4']}) for w,r in rows.items()}
    out.mkdir(parents=True)
    base.write_json(out/'summary.json',dict(coverage=summary,receivers=[asdict(r) for r in receivers],
        pilot_seed=request['pilot_seed'],training_blocks_only=True,elapsed_s=time.monotonic()-started))
    for w,r in rows.items():base.write_jsonl(out/f'{w}km_rows.jsonl',r)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/multireceiver_v7.json')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();request=load(args.config)
    if args.pilot:pilot(request)
    else:
        from multireceiver_campaign_v7 import run
        run(args.config,args.resume)

if __name__=='__main__':main()
