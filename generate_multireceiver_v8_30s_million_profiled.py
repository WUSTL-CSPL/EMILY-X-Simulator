#!/usr/bin/env python3
"""Resume v8 30s with original gzip export and per-batch performance records.

Original simulation, sampling, seeds, validation and checkpoint transactions are
reused. This wrapper adds measurements only; existing scripts are untouched.
"""
import argparse
import functools
import resource
import generate_multireceiver_v8_30s_million as legacy
from generate_multireceiver_v8_30s_million import *

ORIGINAL_INITIALIZE=g.initialize_worker
STAGES={}

def timed(fn,name):
    @functools.wraps(fn)
    def call(*a,**kw):
        start=time.perf_counter()
        try:return fn(*a,**kw)
        finally:STAGES[name]=STAGES.get(name,0.)+time.perf_counter()-start
    return call


def initialize_profiled(freq,models,config,receivers):
    ORIGINAL_INITIALIZE(freq,models,config,receivers)
    base.sim.prepare_propagation=timed(base.sim.prepare_propagation,'propagation')
    g.ORIGINAL_SPECTROGRAM=timed(g.ORIGINAL_SPECTROGRAM,'spectrogram')
    base.sim.run_local_emily_windows=timed(base.sim.run_local_emily_windows,'detector')


def profile_simulate(scenario):
    STAGES.clear();start=time.perf_counter();cpu=time.process_time()
    row=g.simulate(scenario)
    return row,dict(wall_s=time.perf_counter()-start,cpu_s=time.process_time()-cpu,
        rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,stages=dict(STAGES))

def advance(db,old,pool,width,split,target,request,config,receivers,campaign,budget):
    state=get_state(db,width,split)
    stream=stream_for(width,split,request,config,receivers,state)
    blocks={b['block_id']:b for b in stream.blocks}
    while any(state['counts'][str(n)]<target for n in (3,4)):
        if legacy.STOP or (budget is not None and budget[0]<=0):return False
        resource_check(campaign,request)
        count=min(request['batch_size'],request['max_proposals_per_split']-state['proposed'])
        if count<=0:raise RuntimeError('Proposal limit reached; no detection conditions were relaxed.')
        started=time.monotonic();parent_cpu=time.process_time();scenarios=[stream.next() for _ in range(count)]
        proposal_s=time.monotonic()-started;dispatch_start=time.monotonic();encode_s=0.;worker_metrics=[]
        rows={};payloads={};jobs={};new_ids=set()
        for scenario in scenarios:
            sid=scenario['sample_id']
            cached=db.execute('select payload from scenes where id=?',(sid,)).fetchone()
            if cached is None and old is not None:
                cached=old.execute('select payload from scenes where id=?',(sid,)).fetchone()
            if cached is not None:
                payloads[sid]=cached[0];rows[sid]=json.loads(zlib.decompress(cached[0]))
            else:jobs[pool.submit(profile_simulate,scenario)]=scenario
        dispatch_s=time.monotonic()-dispatch_start;collect_start=time.monotonic()
        for future in futures.as_completed(jobs):
            row,metrics=future.result();worker_metrics.append(metrics);sid=row['sample_id'];rows[sid]=row
            t=time.monotonic();payloads[sid]=zlib.compress(encode(row),1);encode_s+=time.monotonic()-t;new_ids.add(sid)
        collect_s=time.monotonic()-collect_start;validation_start=time.monotonic()
        for scenario in scenarios:
            validate_scene(rows[scenario['sample_id']],scenario,width,split,request,blocks,receivers)
        # Cache, selection ranks and exact RNG state are committed together.
        # A crash before commit only repeats this batch; a committed batch is never repeated.
        validation_s=time.monotonic()-validation_start;transaction_start=time.monotonic()
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
        transaction_s=time.monotonic()-transaction_start
        metrics=dict(updated_unix=time.time(),width_km=width,split=split,proposed=state['proposed'],
            new_simulations=len(new_ids),selected=dict(state['counts']),batch_wall_s=time.monotonic()-started,
            parent_cpu_s=time.process_time()-parent_cpu,proposal_s=proposal_s,dispatch_and_cache_read_s=dispatch_s,
            collect_results_s=collect_s,encode_compress_s=encode_s,validate_s=validation_s,transaction_commit_s=transaction_s,
            worker_cpu_s=sum(m['cpu_s'] for m in worker_metrics),worker_wall_s=sum(m['wall_s'] for m in worker_metrics),
            max_worker_rss_kib=max((m['rss_kib'] for m in worker_metrics),default=0),
            physical_stages_s={name:sum(m['stages'].get(name,0.) for m in worker_metrics) for name in ('propagation','spectrogram','detector')},
            load_average=os.getloadavg(),parent_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            system_memory={line.split(':')[0]:line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines() if line.split(':')[0] in ('MemAvailable','SwapTotal','SwapFree')})
        with (campaign/'batch_timings.jsonl').open('a') as handle:handle.write(json.dumps(metrics)+'\n')
        if budget is not None:budget[0]-=1
        base.write_json(campaign/'progress.json',dict(status='running',updated_unix=time.time(),
            width_km=width,split=split,stage_target=target,state=state,minimum_free_gib=request['minimum_free_gib']))
        print(f'{width}km {split}: proposed={state["proposed"]:,}; selected={state["counts"]}; stage_target={target:,}; new={len(new_ids)}; batch_s={time.monotonic()-started:.1f}',flush=True)
    return True


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/multireceiver_v8_30s_million.json')
    args,_=parser.parse_known_args();request=g.load(args.config);campaign=ROOT/request['campaign_dir']
    for n in (3,4):
        out=ROOT/f"{request['output_prefix']}_{n}rx_20km_{request['small_size']}"
        if (out/'config.json').exists():
            assert '.jsonl.gz' in g.load(out/'config.json')['format'],'Existing uncompressed output must not be overwritten'
    original_connect=legacy.connect
    def connect_with_policy(path):
        policy=dict(format='gzip_jsonl',compression_authorized=True,simulation_unchanged=True,
                    original_generator_sha256=base.sha256(legacy.__file__),profiled_wrapper_sha256=base.sha256(__file__))
        policy_path=campaign/'profiled_gzip_export_policy.json'
        if policy_path.exists():assert g.load(policy_path)==policy
        else:
            base.write_json(policy_path,policy)
            shutil.copyfile(__file__,campaign/'code'/Path(__file__).name)
        return original_connect(path)
    legacy.connect=connect_with_policy
    legacy.advance=advance
    g.initialize_worker=initialize_profiled
    legacy.main()


if __name__=='__main__':main()
