#!/usr/bin/env python3
"""Disk-backed, resumable v8 generation with bounded-memory JSONL.gz shards."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import concurrent.futures as futures
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time
import zlib

import generate_multireceiver_dataset_v8 as g

ROOT, base = g.ROOT, g.base
STOP = False


def stopping(*_):
    global STOP
    STOP = True


def encode(value):
    return json.dumps(value,allow_nan=False,separators=(',',':')).encode()


def connect(path):
    db=sqlite3.connect(path)
    db.execute('pragma journal_mode=WAL')
    db.execute('pragma synchronous=FULL')
    db.execute('pragma cache_size=-65536')
    db.executescript('''
        CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY, payload BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS pools(width INTEGER, split TEXT, state TEXT NOT NULL,
          PRIMARY KEY(width,split));
        CREATE TABLE IF NOT EXISTS selections(width INTEGER, split TEXT, n INTEGER,
          rank INTEGER, id TEXT NOT NULL, PRIMARY KEY(width,split,n,rank),
          UNIQUE(width,split,n,id));
        CREATE TABLE IF NOT EXISTS proposals(width INTEGER, split TEXT, ordinal INTEGER,
          id TEXT NOT NULL, mask TEXT NOT NULL, PRIMARY KEY(width,split,ordinal),
          UNIQUE(width,split,id));
    ''')
    return db


def initial_state():
    return dict(proposed=0,counts={'3':0,'4':0},masks={},blocks={},new_simulations=0,
                reused_scenes=0,simulation_seconds=0.,rng=None,counters={})


def get_state(db,width,split):
    record=db.execute('select state from pools where width=? and split=?',(width,split)).fetchone()
    return json.loads(record[0]) if record else initial_state()


def stream_for(width,split,request,config,receivers,state):
    stream=g.Proposals(width,split,request,config,receivers)
    if state['rng'] is not None:
        stream.rng.bit_generator.state=state['rng']
        stream.counters=Counter(state['counters'])
    return stream


def validate_scene(row,scenario,width,split,request,blocks,receivers):
    if any(row['truth'].get(k)!=v for k,v in scenario.items()):
        raise ValueError('Cached scenario differs from deterministic proposal')
    base.sim.RECEIVERS=receivers
    base.validate_row(row)
    g.validate_randomization(row,4,request)
    t=row['truth'];b=blocks[t['block_id']]
    assert t['split']==b['split']==split
    assert abs(t['x_m'])<=width*500 and abs(t['y_m'])<=width*500
    xmin,xmax,ymin,ymax=b['bounds_m'];margin=request['boundary_buffer_m']
    assert min(t['x_m']-xmin,xmax-t['x_m'],t['y_m']-ymin,ymax-t['y_m'])>=margin-1e-6
    assert t['receivers']==[asdict(r) for r in receivers]
    assert set(t['links'])=={r.name for r in receivers}
    assert all(v['propagation_model']=='ITU-R P.452-16 + SRTM' for v in t['links'].values())
    assert all(v['tx_ground_elevation_m']>10 and v['rx_ground_elevation_m']>10 for v in t['links'].values())
    assert min(base.sim.GEOD.inv(t['longitude_deg'],t['latitude_deg'],r.lon_deg,r.lat_deg)[2] for r in receivers)>=request['min_distance_to_receiver_m']


def resource_check(path,request):
    free=shutil.disk_usage(path).free
    if free<request['minimum_free_gib']*1024**3:
        raise RuntimeError(f'Free disk below {request["minimum_free_gib"]} GiB; checkpoint retained. Free space or move the campaign before resuming.')


def advance(db,old,pool,width,split,target,request,config,receivers,campaign,budget):
    state=get_state(db,width,split)
    stream=stream_for(width,split,request,config,receivers,state)
    blocks={b['block_id']:b for b in stream.blocks}
    while any(state['counts'][str(n)]<target for n in (3,4)):
        if STOP or (budget is not None and budget[0]<=0):return False
        resource_check(campaign,request)
        count=min(request['batch_size'],request['max_proposals_per_split']-state['proposed'])
        if count<=0:raise RuntimeError('Proposal limit reached; no detection conditions were relaxed.')
        started=time.monotonic();scenarios=[stream.next() for _ in range(count)]
        rows={};payloads={};jobs={};new_ids=set()
        for scenario in scenarios:
            sid=scenario['sample_id']
            cached=db.execute('select payload from scenes where id=?',(sid,)).fetchone()
            if cached is None and old is not None:
                cached=old.execute('select payload from scenes where id=?',(sid,)).fetchone()
            if cached is not None:
                payloads[sid]=cached[0];rows[sid]=json.loads(zlib.decompress(cached[0]))
            else:jobs[pool.submit(g.simulate,scenario)]=scenario
        for future in futures.as_completed(jobs):
            row=future.result();sid=row['sample_id'];rows[sid]=row
            payloads[sid]=zlib.compress(encode(row),1);new_ids.add(sid)
        for scenario in scenarios:
            validate_scene(rows[scenario['sample_id']],scenario,width,split,request,blocks,receivers)
        # Cache, selection ranks and exact RNG state are committed together.
        # A crash before commit only repeats this batch; a committed batch is never repeated.
        with db:
            for scenario in scenarios:
                sid=scenario['sample_id'];row=rows[sid]
                db.execute('insert or ignore into scenes values(?,?)',(sid,payloads[sid]))
                flags=[bool(r['incidents']) for r in row['input']['receivers']]
                mask=''.join(str(int(x)) for x in flags)
                state['proposed']+=1
                db.execute('insert into proposals values(?,?,?,?,?)',(width,split,state['proposed'],sid,mask))
                state['masks'][mask]=state['masks'].get(mask,0)+1
                b=state['blocks'].setdefault(scenario['block_id'],dict(proposed=0,eligible_3rx=0,eligible_4rx=0))
                b['proposed']+=1
                for n in (3,4):
                    if all(flags[:n]):
                        b[f'eligible_{n}rx']+=1
                        # Keep all eligible prefixes up to the large quota, even in the small stage.
                        if state['counts'][str(n)]<request['large_splits'][split]:
                            state['counts'][str(n)]+=1
                            db.execute('insert into selections values(?,?,?,?,?)',
                                (width,split,n,state['counts'][str(n)],sid))
            state['new_simulations']+=len(new_ids)
            state['reused_scenes']+=len(scenarios)-len(new_ids)
            state['simulation_seconds']+=time.monotonic()-started
            state['rng']=stream.rng.bit_generator.state;state['counters']=dict(stream.counters)
            db.execute('insert or replace into pools values(?,?,?)',(width,split,json.dumps(state)))
        if budget is not None:budget[0]-=1
        base.write_json(campaign/'progress.json',dict(status='running',updated_unix=time.time(),
            width_km=width,split=split,stage_target=target,state=state,minimum_free_gib=request['minimum_free_gib']))
        print(f'{width}km {split}: proposed={state["proposed"]:,}; selected={state["counts"]}; stage_target={target:,}; new={len(new_ids)}; batch_s={time.monotonic()-started:.1f}',flush=True)
    return True


def gzip_writer(path):
    return gzip.GzipFile(filename=str(path),mode='wb',compresslevel=6,mtime=0)


def verify_part(path,expected_count):
    info=g.load(path/'manifest.json')
    assert info['count']==expected_count
    for name,expected in info['sha256'].items():assert base.sha256(path/name)==expected
    with ExitStack() as stack:
        handles=[stack.enter_context(gzip.open(path/(name+'.gz'),'rt')) for name in g.TABLES]
        count=0
        while True:
            lines=[f.readline() for f in handles]
            if not any(lines):break
            assert all(lines)
            values=[json.loads(line) for line in lines]
            assert len({v['sample_id'] for v in values})==1
            count+=1
        assert count==expected_count
    return info


def export_dataset(db,n,width,size,request,config,receivers,manifest,campaign):
    root=ROOT/f"{request['output_prefix']}_{n}rx_{width}km_{size}"
    root.mkdir(parents=True,exist_ok=True)
    sizes=request['small_splits'] if size==request['small_size'] else request['large_splits']
    metadata=dict(signature=manifest['signature'],receiver_count=n,width_km=width,total_samples=size,
        sizes=sizes,receivers=[asdict(r) for r in receivers[:n]],request=request,
        format='Each split has part-*/{inputs,labels,ground_truth}.jsonl.gz. Same JSON fields as v8.',
        selection='First eligible N per split; every included receiver has an incident.',
        small_is_large_prefix=True,common_ids_same_split=True)
    if (root/'config.json').exists():assert g.load(root/'config.json')==metadata
    else:
        base.write_json(root/'config.json',metadata)
        base.write_json(root/'manifest.json',manifest)
        (root/'README.md').write_text('# Large v8 dataset\n\nRead config.json for sizes and generation settings. Each split contains aligned gzip-compressed JSONL shards. Read each part in numeric order, then join the three tables by sample_id. Use inputs and labels for training; ground_truth is audit-only.\n\nOnly COMPLETE.json confirms this dataset is complete. Small versions are prefixes of large versions; do not concatenate them as independent data. The original v8 datasets are also prefixes because the seed and proposal order are unchanged.\n')
    shard_index={};base.sim.RECEIVERS=receivers[:n]
    for split,target in sizes.items():
        available=db.execute('select count(*) from selections where width=? and split=? and n=?',(width,split,n)).fetchone()[0]
        assert available>=target
        directory=root/split;directory.mkdir(exist_ok=True);shard_index[split]=[]
        for start in range(1,target+1,request['shard_samples']):
            if STOP:return False
            resource_check(campaign,request)
            end=min(target,start+request['shard_samples']-1);count=end-start+1
            part=directory/f'part-{(start-1)//request["shard_samples"]:06d}'
            if part.exists():
                info=verify_part(part,count)
                assert info['first_rank']==start and info['last_rank']==end
                shard_index[split].append(dict(path=str(part.relative_to(root)),**info));continue
            temporary=part.with_name(part.name+'.partial')
            if temporary.exists():shutil.rmtree(temporary)  # Only an uncommitted shard owned by this campaign.
            temporary.mkdir();written=0;first=None;last=None;id_hash=hashlib.sha256()
            rows=db.execute('''select s.id,c.payload from selections s join scenes c on c.id=s.id
              where s.width=? and s.split=? and s.n=? and s.rank between ? and ? order by s.rank''',
              (width,split,n,start,end))
            with ExitStack() as stack:
                handles=[stack.enter_context(gzip_writer(temporary/(name+'.gz'))) for name in g.TABLES]
                for sid,payload in rows:
                    row=g.subset_receivers(json.loads(zlib.decompress(payload)),n)
                    base.validate_row(row);g.validate_randomization(row,n,request)
                    assert g.all_detected(row,n) and row['truth']['split']==split
                    values=[{'sample_id':sid,**row['input']},{'sample_id':sid,**row['label']},row['truth']]
                    for handle,value in zip(handles,values):handle.write(encode(value)+b'\n')
                    written+=1;first=first or sid;last=sid;id_hash.update((sid+'\n').encode())
            assert written==count
            info=dict(count=count,first_rank=start,last_rank=end,first_id=first,last_id=last,
                ordered_ids_sha256=id_hash.hexdigest(),sha256={name+'.gz':base.sha256(temporary/(name+'.gz')) for name in g.TABLES})
            base.write_json(temporary/'manifest.json',info)
            verify_part(temporary,count)
            temporary.rename(part)
            shard_index[split].append(dict(path=str(part.relative_to(root)),**info))
            print(f'Exported {root.name}/{split}/{part.name}: {count:,} samples',flush=True)
    base.write_json(root/'shards.json',shard_index)
    base.write_json(root/'selection_report.json',{s:get_state(db,width,s) for s in sizes})
    base.write_json(root/'COMPLETE.json',dict(signature=manifest['signature'],samples=size,sizes=sizes,
        shards_verified=True,all_receivers_detected=True,minimum_geographic_buffer_between_splits_m=2*request['boundary_buffer_m']))
    return True


def run(args):
    request=g.load(args.config);campaign=ROOT/request['campaign_dir']
    assert sum(request['small_splits'].values())==request['small_size']
    assert sum(request['large_splits'].values())==request['large_size']
    assert all(0<request['small_splits'][s]<=request['large_splits'][s] for s in g.SPLITS)
    assert request['batch_size']>0 and request['workers']>0 and request['shard_samples']>0
    if campaign.exists() and not args.resume:raise FileExistsError('Use --resume for an existing campaign')
    campaign.mkdir(parents=True,exist_ok=True)
    lock=(campaign/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=g.load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    source_manifest=g.load(ROOT/'runs/multireceiver_v8/manifest.json')
    assert base.sha256(base.__file__)==config['generator_sha256']
    assert base.sha256(base.SIM_SOURCE)==config['simulator_sha256']
    assert base.sha256(ROOT/'EMILY-X/srsran_loaded.cf32')==config['iq_sha256']
    assert base.sha256(g.__file__)==source_manifest['signature_data']['code_sha256'][Path(g.__file__).name]
    runtime=base.runtime_manifest(config)
    runtime['scope']='Original v8 physical chain, random 33-43 dBm TX power and per-receiver Gaussian measurement offsets.'
    for field in ('python','packages','terrain_sha256','simulator_git_commit'):
        assert runtime[field]==source_manifest[field],f'Changed v8 dependency: {field}'
    original_request=g.load(ROOT/'configs/multireceiver_v8.json')
    for field in ('sampling_seed','extension_split_seed','boundary_buffer_m','min_distance_to_receiver_m',
                  'added_receivers','transmitter_power_dbm_min','transmitter_power_dbm_max','measurement_offset_sigma_db'):
        assert request[field]==original_request[field],f'Changed v8 setting: {field}'
    sources=[Path(__file__),Path(g.__file__),Path(base.__file__),base.SIM_SOURCE]
    manifest=dict(request=request,source_config=config,runtime=runtime,
                  code_sha256={p.name:base.sha256(p) for p in sources},original_v8_signature=source_manifest['signature'])
    manifest['signature']=hashlib.sha256(encode(manifest)).hexdigest()
    if (campaign/'manifest.json').exists():assert g.load(campaign/'manifest.json')==manifest
    else:
        for n in (3,4):
            for width,size in [(20,request['small_size']),(20,request['large_size']),(40,request['large_size'])]:
                assert not (ROOT/f"{request['output_prefix']}_{n}rx_{width}km_{size}").exists(),'Refusing to adopt existing output'
        base.write_json(campaign/'manifest.json',manifest)
        snapshots=campaign/'code';snapshots.mkdir()
        for p in sources:shutil.copyfile(p,snapshots/p.name)
        protected={str(p):base.sha256(p) for p in ROOT.glob('*v8.py')}
        protected.update({str(p):base.sha256(p) for p in (ROOT/'configs').glob('*v8.json')})
        protected.update({str(p):base.sha256(p) for p in (ROOT/'datasets').glob('terrestrial_location_v8_*.zip')})
        base.write_json(campaign/'protected_originals.json',protected)
    protected=g.load(campaign/'protected_originals.json')
    assert all(base.sha256(p)==sha for p,sha in protected.items())
    resource_check(campaign,request)
    db=connect(campaign/'scene_cache.sqlite');old=None
    if request.get('reuse_v8_cache',True):
        old=sqlite3.connect('file:'+str(ROOT/'runs/multireceiver_v8/scene_cache.sqlite')+'?mode=ro',uri=True)
    receivers=g.receivers_for(request,config);freq,models=base.build_models()
    budget=[args.max_batches] if args.max_batches is not None else None
    base.write_json(campaign/'process.json',dict(pid=os.getpid(),started_unix=time.time(),signature=manifest['signature']))
    base.write_json(campaign/'status.json',dict(status='running',updated_unix=time.time()))
    stages=[(20,request['small_size'],request['small_splits'])]
    if not request.get('only_small',False):
        stages += [(20,request['large_size'],request['large_splits']),(40,request['large_size'],request['large_splits'])]
    try:
        with futures.ProcessPoolExecutor(max_workers=request['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=g.initialize_worker,initargs=(freq,models,config,receivers)) as pool:
            # Finish the two million-sample datasets first, then extend their pools.
            for width,size,sizes in stages:
                for split in g.SPLITS:
                    if not advance(db,old,pool,width,split,sizes[split],request,config,receivers,campaign,budget):
                        base.write_json(campaign/'status.json',dict(status='checkpointed',reason='signal or batch limit',updated_unix=time.time()))
                        return
                for n in (3,4):
                    if not export_dataset(db,n,width,size,request,config,receivers,manifest,campaign):
                        base.write_json(campaign/'status.json',dict(status='checkpointed',reason='signal during export',updated_unix=time.time()))
                        return
        assert all(base.sha256(p)==sha for p,sha in protected.items())
        base.write_json(campaign/'COMPLETE.json',dict(status='complete',signature=manifest['signature'],
            unique_cached_scenes=db.execute('select count(*) from scenes').fetchone()[0],
            total_exported_samples=2*sum(size for _,size,_ in stages),
            completed_datasets=[str(ROOT/f"{request['output_prefix']}_{n}rx_{width}km_{size}") for width,size,_ in stages for n in (3,4)],
            all_requested_datasets_complete=True,originals_unchanged=True,finished_unix=time.time()))
        base.write_json(campaign/'status.json',dict(status='complete',updated_unix=time.time()))
    except BaseException as exc:
        base.write_json(campaign/'status.json',dict(status='failed',error=repr(exc),updated_unix=time.time(),resume_supported=True))
        raise
    finally:
        db.close()
        if old is not None:old.close()
        lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/multireceiver_v8_large.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-batches',type=int,help='Stop cleanly after this many committed batches (for restart checks).')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    run(args)


if __name__=='__main__':main()
