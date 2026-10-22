"""Resumable proposal pool and exports for the six v7 datasets."""
from dataclasses import asdict
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import time
import zipfile
import zlib
from collections import Counter
import concurrent.futures as futures
import multiprocessing

import generate_multireceiver_dataset_v7 as g
from scipy.spatial import cKDTree

ROOT,base,np=g.ROOT,g.base,g.np


def cache_get(db,sid):
    row=db.execute('select payload from scenes where id=?',(sid,)).fetchone()
    return json.loads(zlib.decompress(row[0])) if row else None


def existing_files():
    paths=list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+[base.SIM_SOURCE]
    for directory in ['configs','docs','tests','scripts','datasets']:
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and not any('v7' in part for part in p.parts)]
    return sorted(set(paths))


def baseline_checks(freq,models,config,receivers):
    checks=[]
    for split in g.SPLITS:
        source=ROOT/'datasets/terrestrial_location_v3'/split
        old_truth=json.loads((source/'ground_truth.jsonl').open().readline())
        old_input=json.loads((source/'inputs.jsonl').open().readline())
        keys=('sample_id','x_m','y_m','latitude_deg','longitude_deg','cell_x','cell_y','receiver_seeds','geometry_replacements')
        scenario={k:old_truth[k] for k in keys}
        base.initialize_worker(freq,models,config)
        row=base.simulate_scene(scenario)
        assert row['truth']==old_truth and {'sample_id':row['sample_id'],**row['input']}==old_input
        # Add receivers without changing the original receiver noise streams.
        scenario={**scenario,'receiver_seeds':scenario['receiver_seeds']+[7103,7104]}
        g.initialize_worker(freq,models,config,receivers)
        multi=g.simulate(scenario)
        assert g.subset_receivers(multi,2)==row
        checks.append(dict(split=split,sample_id=row['sample_id'],original_2rx_exact=True,
                           adding_receivers_preserves_original_observations=True))
    return checks


def pool_for(width,split,target,request,config,receivers,db,pool,campaign):
    stream=g.Proposals(width,split,request,config,receivers)
    selected={3:[],4:[]};counts=Counter();by_block={};seen=0;new=0;started=time.monotonic()
    audit_path=campaign/f'proposals_{width}km_{split}.csv'
    with audit_path.open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['sample_id','block_id','region','latitude_deg','longitude_deg','RX1','RX2','RX3','RX4','eligible_3rx','eligible_4rx','selected_3rx','selected_4rx'])
        while len(selected[4])<target or len(selected[3])<target:
            batch_size=min(request['batch_size'],request['max_proposals_per_split']-seen)
            if batch_size<=0: raise RuntimeError(f'Proposal budget exhausted at {width}/{split}; do not change detector or silently relax selection')
            scenarios=[stream.next() for _ in range(batch_size)]
            rows={};jobs={}
            for scenario in scenarios:
                row=cache_get(db,scenario['sample_id'])
                if row is None: jobs[pool.submit(g.simulate,scenario)]=scenario
                else:
                    assert all(row['truth'][k]==v for k,v in scenario.items())
                    rows[scenario['sample_id']]=row
            for job in futures.as_completed(jobs):
                scenario=jobs[job];row=job.result()
                assert all(row['truth'][k]==v for k,v in scenario.items())
                payload=zlib.compress(json.dumps(row,allow_nan=False).encode(),1)
                db.execute('insert into scenes(id,payload) values(?,?)',(row['sample_id'],payload))
                rows[row['sample_id']]=row;new+=1
            db.commit()
            for scenario in scenarios:
                row=rows[scenario['sample_id']];sid=row['sample_id']
                flags=tuple(bool(r['incidents']) for r in row['input']['receivers'])
                eligible=[all(flags[:3]),all(flags)]
                retained=[]
                for n,good in zip((3,4),eligible):
                    take=good and len(selected[n])<target
                    if take:selected[n].append(sid)
                    retained.append(take)
                mask=''.join(str(int(f)) for f in flags);counts[mask]+=1
                b=by_block.setdefault(scenario['block_id'],dict(proposed=0,eligible_3rx=0,eligible_4rx=0))
                b['proposed']+=1;b['eligible_3rx']+=eligible[0];b['eligible_4rx']+=eligible[1]
                writer.writerow([sid,scenario['block_id'],scenario['region'],scenario['latitude_deg'],scenario['longitude_deg'],
                                 *map(int,flags),*map(int,eligible),*map(int,retained)])
            seen+=len(scenarios);handle.flush()
            progress=dict(width_km=width,split=split,proposed=seen,new_simulations=new,
                          selected={str(n):len(v) for n,v in selected.items()},elapsed_s=time.monotonic()-started)
            base.write_json(campaign/'progress.json',progress)
            print(f'{width}km {split}: {seen} proposals, accepted 3rx={len(selected[3])}/{target}, 4rx={len(selected[4])}/{target}, new={new}, {progress["elapsed_s"]:.1f}s',flush=True)
    report=dict(proposals=seen,new_simulations=new,cached_reuses=seen-new,detection_masks=dict(counts),
        eligible_3rx=sum(v for k,v in counts.items() if k.startswith('111')),
        eligible_4rx=counts['1111'],per_block=by_block,elapsed_s=time.monotonic()-started,
        note='Counts include the entire final batch; first eligible N per receiver count are selected in proposal order.')
    base.write_json(campaign/f'pool_{width}km_{split}.json',report)
    return selected,report


def validate_export(rows_by_split,n,request,config,receivers):
    g.base.sim.RECEIVERS=receivers[:n]
    ids=set();coords=set();xy={};coverage={}
    forward,_=base.projection(config['profile'])
    roles=g.load(ROOT/'datasets/terrestrial_location_v3/split_manifest.json')['block_roles']
    # The 40 km grid contains the same core rectangles and split assignments.
    blocks={b['block_id']:b for b in g.blocks_for(40,roles,request['extension_split_seed'])}
    for split,rows in rows_by_split.items():
        xy[split]=[];regions=Counter();hist=Counter();incidents=0
        for row in rows:
            base.validate_row(row)
            assert len(row['input']['receivers'])==n and g.all_detected(row,n)
            assert row['truth']['receivers']==[asdict(r) for r in receivers[:n]]
            assert set(row['truth']['links'])=={r.name for r in receivers[:n]}
            assert len(row['truth']['receiver_seeds'])==n
            sid=row['sample_id'];t=row['truth'];position=(t['latitude_deg'],t['longitude_deg'])
            assert sid not in ids and position not in coords;ids.add(sid);coords.add(position)
            assert t['split']==split==blocks[t['block_id']]['split']
            xmin,xmax,ymin,ymax=blocks[t['block_id']]['bounds_m']
            assert min(t['x_m']-xmin,xmax-t['x_m'],t['y_m']-ymin,ymax-t['y_m'])>=request['boundary_buffer_m']-1e-6
            assert np.allclose(forward.transform(t['longitude_deg'],t['latitude_deg']),[t['x_m'],t['y_m']],atol=1e-5,rtol=0)
            assert min(base.sim.GEOD.inv(t['longitude_deg'],t['latitude_deg'],r.lon_deg,r.lat_deg)[2] for r in receivers)>=request['min_distance_to_receiver_m']
            assert all(v['propagation_model']=='ITU-R P.452-16 + SRTM' for v in t['links'].values())
            assert all(v['tx_ground_elevation_m']>10 and v['rx_ground_elevation_m']>10 for v in t['links'].values())
            for group in row['input']['receivers']:
                incidents+=len(group['incidents'])
                for inc in group['incidents']:
                    if inc['ongoing']:assert inc['duration_sec'] is None and inc['t_end'] is None
            xy[split].append([t['x_m'],t['y_m']]);regions[t['region']]+=1;hist[t['block_id']]+=1
        coverage[split]=dict(samples=len(rows),all_receivers_detected=len(rows),incidents=incidents,regions=dict(regions),per_block=dict(hist))
    distances={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        distances[f'{a}_to_{b}']=float(cKDTree(xy[a]).query(xy[b])[0].min())
        assert distances[f'{a}_to_{b}']>=500-1e-6
    return coverage,distances


def dataset_name(request,n,width,size):return f"{request['output_prefix']}_{n}rx_{width}km_{size}"


def export_dataset(n,width,size,selected,reports,db,request,config,receivers,manifest,replays,resume):
    out=ROOT/dataset_name(request,n,width,size);sizes=request[f'sizes_{size}']
    if out.exists() and not resume:raise FileExistsError(out)
    if out.exists():assert g.load(out/'manifest.json')['signature']==manifest['signature']
    out.mkdir(parents=True,exist_ok=True)
    base.write_json(out/'manifest.json',manifest)
    rows={split:[g.subset_receivers(cache_get(db,sid),n) for sid in selected[split][n][:sizes[split]]] for split in g.SPLITS}
    coverage,distances=validate_export(rows,n,request,config,receivers)
    for split,items in rows.items():base.export_rows(out/split,items)
    resolved=dict(receiver_count=n,width_km=width,total_samples=size,sizes=sizes,
        receivers=[asdict(r) for r in receivers[:n]],simulation={**config,'receivers':[asdict(r) for r in receivers[:n]],
        'count':size,'width_km':width,'cells':None,'incident_training_target':0,'sampling':'v7 geographically buffered proposals conditioned on detection at every included receiver',
        'rejection_sampling':True},request=request,
        selection='First N proposals per geographic split with incidents at every included receiver. Small datasets are per-split prefixes of the large datasets.',
        common_geometry_exclusion='All proposals remain >=1 km from all four candidate sites, including RX4 for the 3-receiver datasets.')
    base.write_json(out/'config.json',resolved)
    roles=g.load(ROOT/'datasets/terrestrial_location_v3/split_manifest.json')['block_roles']
    base.write_json(out/'split_manifest.json',dict(blocks=g.blocks_for(width,roles,request['extension_split_seed']),
        sample_ids={s:[r['sample_id'] for r in v] for s,v in rows.items()},sizes=sizes,boundary_buffer_m=250,
        minimum_cross_split_distance_m=distances))
    base.write_json(out/'selection_report.json',dict(pool=reports,
        acceptance_fraction={s:reports[s][f'eligible_{n}rx']/reports[s]['proposals'] for s in g.SPLITS},
        selection='Detection-conditioned; not uniform coverage of the geographic square. Same proposal pool supplies 3rx/4rx and small/large sets.',
        source_audits='Full proposed positions, detection masks and rejected records remain in campaign CSVs and SQLite cache.'))
    base.write_json(out/'summary.json',dict(splits=coverage,all_samples_detected_at_every_receiver=True,model_training_performed=False))
    # Exact replays for first/last scenes per split: cache reuse cannot mask seed bugs.
    repeat=[]
    for split in g.SPLITS:
        for row in [rows[split][0],rows[split][-1]]:
            original=cache_get(db,row['sample_id'])
            keys=('sample_id','x_m','y_m','latitude_deg','longitude_deg','cell_x','cell_y','block_id','split','region','proposal_index_in_block','geometry_replacements','receiver_seeds')
            scenario={k:original['truth'][k] for k in keys}
            base.sim.RECEIVERS=receivers
            assert g.simulate(scenario)==original
            repeat.append(row['sample_id'])
    base.write_json(out/'verification.json',dict(passed=True,all_rows_validated=True,all_receivers_have_incidents=True,
        unique_ids_and_positions=True,geographic_separation_passed=True,original_replays=replays,
        exact_fresh_replays=repeat,table_sha256={f'{s}/{f}':base.sha256(out/s/f) for s in g.SPLITS for f in g.TABLES}))
    code=out/'code';code.mkdir(exist_ok=True)
    for file in [Path(g.__file__),Path(__file__),Path(base.__file__)]:shutil.copyfile(file,code/file.name)
    lines=[f'# v7: {n} omnidirectional receivers, {width} x {width} km, {size:,} scenes','',
        'Every exported scene has at least one incident at every included receiver. RX1/RX2 keep their original positions. RX3/RX4 are fixed added sites.',
        'Receivers: 0 dBi omnidirectional gain, 2 m height, 6 dB noise figure. The original transmitter sector pattern remains unchanged.',
        'Same loaded LTE waveform, 800 MHz carrier, 10 MHz bandwidth, 43 dBm conducted power, 40 m transmitter height and 60-second observation window.',
        'Full upstream P.452/SRTM propagation and local EMILY detector are reused. No directional receiver adapter is imported.', '',
        '## Files and training','',
        'Each split has inputs.jsonl, labels.jsonl and ground_truth.jsonl joined by sample_id. One scene contains all receiver incident lists.',
        'Use inputs as features and source latitude/longitude in labels as targets. Ground truth is audit-only. Incident lat/lon are receiver positions.',
        'Intensity is mW; 10*log10(mW) gives dBm. Null duration/end values on ongoing incidents are censored boundaries, not zero duration.',
        'This supports multi-receiver, signal-strength-based localization. Incidents contain no angle-of-arrival or precision time-of-arrival measurements.',
        'Train/test geographic regions remain separate. Fit scalers only on training data; do not use ground-truth distances, bearings, path losses or sample IDs as features.', '',
        '## Sampling and limits','',
        'Proposals are drawn inside fixed geographic split blocks in proportion to usable block area, then uniformly within each block.',
        'A 250 m inside-edge buffer gives at least 500 m between splits. Proposals stay >=1 km from all four candidate sites for a common sampling domain.',
        'Only scenes detected at every included receiver are retained. This conditions the location distribution on coverage; low-coverage parts of the square are underrepresented or absent.',
        'We do not rebalance final counts per block or change propagation to fill coverage gaps. Selection reports give rejection rates and per-block coverage.',
        'The 1,000-scene datasets are subsets of their corresponding 10,000-scene versions, including validation/test subsets. These are alternative experiments, not independent datasets to concatenate.',
        'The 3rx and 4rx datasets use the same proposal pool but different detection conditions. Use common_validation_test.json for matched-position comparisons.', '',
        '## Reproduce','',
        '```bash','../.venv/bin/python generate_multireceiver_dataset_v7.py --config configs/multireceiver_v7.json','```','',
        'Existing outputs are protected. --resume requires the same source/data/config signature and reuses cached simulations.',
        'Large IQ and terrain files are local dependencies and are not included in this archive.',
        'See config.json for fixed site coordinates and exact parameters, selection_report.json for proposal acceptance, and verification.json for checks.','']
    (out/'README.md').write_text('\n'.join(lines))
    return out,{s:[r['sample_id'] for r in v] for s,v in rows.items()}


def run(config_path,resume=False):
    started=time.monotonic();request=g.load(config_path)
    config=g.load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    assert base.sha256(base.SIM_SOURCE)==config['simulator_sha256']
    assert base.sha256(base.__file__)==config['generator_sha256']
    assert base.sha256(ROOT/'EMILY-X/srsran_loaded.cf32')==config['iq_sha256']
    receivers=g.receivers_for(request,config)
    campaign=ROOT/request['campaign_dir']
    outputs=[ROOT/dataset_name(request,n,w,z) for n in (3,4) for w,z in [(20,1000),(20,10000),(40,10000)]]
    if not resume and (campaign.exists() or any(p.exists() or p.with_suffix('.zip').exists() for p in outputs)):
        raise FileExistsError('Campaign/output already exists; choose fresh config paths or use --resume with identical inputs')
    manifest=base.runtime_manifest(config)
    manifest['random_streams']=dict(proposal_block='[sampling_seed,701,width,split_index]',position='[sampling_seed,702,ix+100,iy+100,k]',
        receiver_noise='[sampling_seed,703,ix+100,iy+100,k] generates four seeds',template_schedule='[20260921,40] unchanged')
    signature_data=dict(request=request,receivers=[asdict(r) for r in receivers],source_config=config,
        runtime=json.loads(json.dumps(manifest)),code_sha256={p.name:base.sha256(p) for p in [Path(g.__file__),Path(__file__),Path(base.__file__),base.SIM_SOURCE]})
    signature=base.hashlib.sha256(json.dumps(signature_data,sort_keys=True).encode()).hexdigest()
    manifest.update(signature=signature,signature_data=signature_data)
    campaign.mkdir(parents=True,exist_ok=True)
    if (campaign/'manifest.json').exists():assert g.load(campaign/'manifest.json')==manifest
    else:base.write_json(campaign/'manifest.json',manifest)
    protected_path=campaign/'protected_files.json'
    if protected_path.exists(): protected=g.load(protected_path)
    else:
        protected={str(p):base.sha256(p) for p in existing_files()};base.write_json(protected_path,protected)
    assert all(base.sha256(p)==h for p,h in protected.items())
    db=sqlite3.connect(campaign/'scene_cache.sqlite')
    db.execute('pragma journal_mode=WAL');db.execute('create table if not exists scenes(id text primary key,payload blob not null)')
    freq,models=base.build_models()
    replays=baseline_checks(freq,models,config,receivers)
    print('Three original two-receiver replays and four-receiver extension checks passed.',flush=True)
    ids_by_dataset={};reports_by_width={}
    try:
        with futures.ProcessPoolExecutor(max_workers=request['workers'],mp_context=multiprocessing.get_context('spawn'),
                initializer=g.initialize_worker,initargs=(freq,models,config,receivers)) as pool:
            for width in (20,40):
                selected={};reports={}
                for split in g.SPLITS:
                    selected[split],reports[split]=pool_for(width,split,request['sizes_10000'][split],request,config,receivers,db,pool,campaign)
                reports_by_width[width]=reports
                g.initialize_worker(freq,models,config,receivers)
                for n in (3,4):
                    for size in ([1000,10000] if width==20 else [10000]):
                        out,ids=export_dataset(n,width,size,selected,reports,db,request,config,receivers,manifest,replays,resume)
                        ids_by_dataset[out.name]=ids
                        print(f'Exported {out.name}',flush=True)
        for name,ids in ids_by_dataset.items():
            common={other:{s:sorted(set(ids[s])&set(other_ids[s])) for s in ['validation','test']}
                    for other,other_ids in ids_by_dataset.items() if other!=name}
            out=next(p for p in outputs if p.name==name)
            base.write_json(out/'common_validation_test.json',common)
            archive=out.with_suffix('.zip')
            temporary=archive.with_suffix('.zip.tmp')
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.rglob('*')):
                    if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
            with zipfile.ZipFile(temporary) as z:assert z.testzip() is None
            temporary.replace(archive)
        assert all(base.sha256(p)==h for p,h in protected.items())
        base.write_json(campaign/'completion.json',dict(passed=True,datasets=[str(p) for p in outputs],
            unique_simulated_scenes=db.execute('select count(*) from scenes').fetchone()[0],
            elapsed_s=time.monotonic()-started,original_files_unchanged=len(protected),
            archive_sha256={p.name:base.sha256(p.with_suffix('.zip')) for p in outputs},
            proposal_reports=reports_by_width))
        print(f'All six datasets complete in {time.monotonic()-started:.1f}s.',flush=True)
    finally:db.close()
