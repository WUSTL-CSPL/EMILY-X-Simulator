#!/usr/bin/env python3
"""Paired observation-length experiment using the unchanged v8 physical model."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import concurrent.futures as futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import time
import zipfile
import zlib

import generate_multireceiver_dataset_v8 as g

ROOT,base,np=g.ROOT,g.base,g.np
VARIANTS=('native_20','native_30','native_60','prefix_20','prefix_30')
STATE={}
RAW=g.ORIGINAL_SPECTROGRAM
PREPARE=base.sim.prepare_propagation


def prep(rx,*args,**kwargs):
    if STATE['prefix']:return STATE['prop60'][rx.name]
    start=time.perf_counter();value=PREPARE(rx,*args,**kwargs)
    STATE['propagation_s']+=time.perf_counter()-start
    if STATE['duration']==60:STATE['prop60'][rx.name]=value
    return value


def spectrum(rx,*args,**kwargs):
    start=time.perf_counter()
    if STATE['prefix']:
        spec,original=STATE['spec60'][rx.name];d=STATE['duration'];truth=deepcopy(original)
        truth['shadowing_db_t']=truth['shadowing_db_t'][:d]
        for comp in truth['component_truth']:
            comp['tx_total_power_dbm_t']=comp['tx_total_power_dbm_t'][:d]
            comp['rx_total_component_dbm_t']=comp['rx_total_component_dbm_t'][:d]
            comp['median_tx_total_power_dbm']=float(np.nanmedian(comp['tx_total_power_dbm_t']))
            comp['median_rx_total_component_dbm']=float(np.nanmedian(comp['rx_total_component_dbm_t']))
        value=(spec[:d],truth)
    else:
        value=RAW(rx,*args,**kwargs)
        if STATE['duration']==60:STATE['spec60'][rx.name]=value
    STATE['spectrum_s']+=time.perf_counter()-start
    return value


def initialize(freq,models,config,receivers):
    STATE.clear();STATE.update(freq=freq,models=models,config=config,receivers=receivers)
    base.sim.prepare_propagation=prep;g.ORIGINAL_SPECTROGRAM=spectrum


def run_one(scenario,duration,prefix=False):
    STATE.update(duration=duration,prefix=prefix,propagation_s=0.,spectrum_s=0.)
    c={**STATE['config'],'duration_s':duration}
    g.initialize_worker(STATE['freq'],STATE['models'],c,STATE['receivers'])
    start=time.perf_counter();row=g.simulate(scenario);elapsed=time.perf_counter()-start
    for r in row['input']['receivers']:
        for inc in r['incidents']:
            assert inc['n_time_samples']==duration
    return row,dict(total_s=elapsed,propagation_s=STATE['propagation_s'],spectrum_s=STATE['spectrum_s'],
                    detector_and_records_s=elapsed-STATE['propagation_s']-STATE['spectrum_s'])


def bundle(scenario):
    STATE.update(spec60={},prop60={})
    rows={};times={}
    durations=[20,30,60];offset=int(scenario['sample_id'][-2:],16)%3
    durations=durations[offset:]+durations[:offset]
    for d in durations:rows[f'native_{d}'],times[f'native_{d}']=run_one(scenario,d)
    for d in (20,30):rows[f'prefix_{d}'],_=run_one(scenario,d,True)
    for name,row in rows.items():
        row['truth']['observation_duration_requested_s']=int(name.split('_')[1])
        row['truth']['duration_experiment_mode']=name.split('_')[0]
    STATE['spec60'].clear();STATE['prop60'].clear()
    return dict(sample_id=scenario['sample_id'],rows=rows,timing=times)


def benchmark_one(task):
    scenario,d=task;STATE.update(spec60={},prop60={})
    row,timing=run_one(scenario,d)
    STATE['spec60'].clear();STATE['prop60'].clear()
    return dict(timing=timing,incidents=sum(len(r['incidents']) for r in row['input']['receivers']))


def export(selected,db,request,config,receivers,manifest,out):
    for n in (3,4):
        for variant in VARIANTS:
            root=out/f'{variant}_{n}rx_1000';root.mkdir(parents=True,exist_ok=True)
            all_rows={}
            base.sim.RECEIVERS=receivers[:n]
            for split,ids in selected.items():
                rows=[]
                for sid in ids:
                    data=json.loads(zlib.decompress(db.execute('select payload from bundles where id=?',(sid,)).fetchone()[0]))
                    row=g.subset_receivers(data['rows'][variant],n)
                    base.validate_row(row);g.validate_randomization(row,n,request)
                    assert g.all_detected(row,n)
                    rows.append(row)
                base.export_rows(root/split,rows);all_rows[split]=len(rows)
            base.write_json(root/'config.json',dict(variant=variant,receiver_count=n,sizes=all_rows,
                receivers=[asdict(r) for r in receivers[:n]],simulation={**config,'duration_s':int(variant.split('_')[1])},request=request,
                selection='Same 1000 positions in every version; all four receivers detect in all five duration/mode variants.'))
            base.write_json(root/'verification.json',dict(all_rows_validated=True,all_receivers_detect=True,
                hashes={f'{s}/{f}':base.sha256(root/s/f) for s in g.SPLITS for f in g.TABLES}))
            if variant.startswith('native'):
                with zipfile.ZipFile(root.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
                    for p in sorted(root.rglob('*')):
                        if p.is_file():z.write(p,str(Path(root.name)/p.relative_to(root)))
                with zipfile.ZipFile(root.with_suffix('.zip')) as z:assert z.testzip() is None
    base.write_json(out/'sample_ids.json',selected)
    base.write_json(out/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
    request=g.load(ROOT/'configs/multireceiver_v8.json')
    config=g.load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    campaign=ROOT/'runs/v8_duration_comparison';out=ROOT/'datasets/v8_duration_comparison'
    if campaign.exists() and not args.resume:raise FileExistsError(campaign)
    campaign.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
    manifest=dict(config=config,request=request,code_sha256=base.sha256(__file__),v8_code_sha256=base.sha256(g.__file__),
        runtime=base.runtime_manifest(config),design='Paired native durations and exact prefixes; all-receiver intersection cohort; fixed geographic splits.')
    if (campaign/'manifest.json').exists():assert g.load(campaign/'manifest.json')==manifest
    else:base.write_json(campaign/'manifest.json',manifest)
    protected_path=campaign/'protected_originals.json'
    if not protected_path.exists():
        paths=[Path(g.__file__),Path(base.__file__),base.SIM_SOURCE,ROOT/'configs/multireceiver_v8.json']
        paths+=list((ROOT/'datasets').glob('terrestrial_location_v8_*.zip'))
        base.write_json(protected_path,{str(p):base.sha256(p) for p in paths})
    protected=g.load(protected_path);assert all(base.sha256(p)==h for p,h in protected.items())
    receivers=g.receivers_for(request,config);freq,models=base.build_models()
    db=sqlite3.connect(campaign/'bundles.sqlite');db.execute('pragma journal_mode=WAL')
    db.execute('create table if not exists bundles(id text primary key,payload blob not null)')
    selected={};audit={};bench_scenarios=[]
    with futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize,initargs=(freq,models,config,receivers)) as pool:
        for split,target in request['sizes_1000'].items():
            stream=g.Proposals(20,split,request,config,receivers);proposals=[];eligible=[]
            minimum={'train':980,'validation':210,'test':210}[split]
            while len(proposals)<minimum or len(eligible)<target:
                count=min(64,max(minimum-len(proposals),64))
                scenes=[stream.next() for _ in range(count)];results={};jobs={}
                for scene in scenes:
                    old=db.execute('select payload from bundles where id=?',(scene['sample_id'],)).fetchone()
                    if old:results[scene['sample_id']]=json.loads(zlib.decompress(old[0]))
                    else:jobs[pool.submit(bundle,scene)]=scene
                for f in futures.as_completed(jobs):
                    value=f.result();results[value['sample_id']]=value
                    db.execute('insert into bundles values(?,?)',(value['sample_id'],zlib.compress(json.dumps(value,allow_nan=False).encode(),1)))
                db.commit()
                for scene in scenes:
                    value=results[scene['sample_id']]
                    if all(g.all_detected(value['rows'][v],4) for v in VARIANTS):eligible.append(scene['sample_id'])
                    proposals.append(scene['sample_id'])
                    if split=='train' and len(bench_scenarios)<128:bench_scenarios.append(scene)
                print(f'{split}: {len(proposals)} proposals; {len(eligible)}/{target} common all-four detections',flush=True)
                base.write_json(campaign/'progress.json',dict(split=split,proposed=len(proposals),joint_detected=len(eligible),target=target))
                if len(proposals)>10000:raise RuntimeError('Insufficient joint coverage')
            selected[split]=eligible[:target];audit[split]=proposals
        benchmark=[]
        for repeat,order in enumerate(((20,60,30),(30,60,20))):
            for d in order:
                start=time.perf_counter();results=list(pool.map(benchmark_one,[(s,d) for s in bench_scenarios]));elapsed=time.perf_counter()-start
                benchmark.append(dict(repeat=repeat,duration_s=d,count=len(results),wall_s=elapsed,
                    scenes_per_second=len(results)/elapsed,timing=results))
                print(f'Benchmark {d}s, repeat {repeat}: {len(results)} scenes in {elapsed:.2f}s',flush=True)
    # A fresh 60-second full-prefix replay must equal the original v8 output.
    old=sqlite3.connect('file:'+str(ROOT/'runs/multireceiver_v8/scene_cache.sqlite')+'?mode=ro',uri=True)
    replays=[]
    for split,ids in selected.items():
        sid=ids[0];data=json.loads(zlib.decompress(db.execute('select payload from bundles where id=?',(sid,)).fetchone()[0]))
        row=deepcopy(data['rows']['native_60']);row['truth'].pop('observation_duration_requested_s');row['truth'].pop('duration_experiment_mode')
        original=json.loads(zlib.decompress(old.execute('select payload from scenes where id=?',(sid,)).fetchone()[0]))
        assert row==original;replays.append(sid)
    old.close()
    export(selected,db,request,config,receivers,manifest,out)
    base.write_json(out/'proposal_ids.json',audit);base.write_json(out/'benchmark.json',benchmark)
    assert all(base.sha256(p)==h for p,h in protected.items())
    base.write_json(out/'generation_verification.json',dict(passed=True,old_60s_exact_replays=replays,protected_files_unchanged=True,
        source_positions=sum(map(len,selected.values())),proposal_count=sum(map(len,audit.values())),all_10_exports_same_ids=True))
    db.close()
    print('Generation complete.',flush=True)

if __name__=='__main__':main()
