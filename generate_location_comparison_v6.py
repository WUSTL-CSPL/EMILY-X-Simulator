#!/usr/bin/env python3
"""Generate paired 20/40 km localization datasets using the unchanged v5 physics."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as futures
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile

import generate_real_antenna_dataset_v5 as v5

base, np = v5.base, v5.np
ROOT = Path(__file__).resolve().parent
SPLITS, TABLES = v5.SPLITS, v5.TABLES


def load(path):
    return json.loads(Path(path).read_text())


def make_blocks(width_km, old_roles, extension_seed):
    """Keep the original 4 km lattice anchored at (-10, -10) km.

    A centered 40 km square clips outer lattice cells to 2 km edge strips.
    Moving the lattice would change the original held-out regions.
    """
    if width_km not in (20, 40):
        raise ValueError('Supported paired widths are 20 and 40 km')
    half, step, anchor = width_km * 500., 4000., -10000.
    blocks = []
    for iy in range(math.floor((-half-anchor)/step), math.ceil((half-anchor)/step)):
        for ix in range(math.floor((-half-anchor)/step), math.ceil((half-anchor)/step)):
            bounds = [max(-half, anchor+ix*step), min(half, anchor+(ix+1)*step),
                      max(-half, anchor+iy*step), min(half, anchor+(iy+1)*step)]
            core = 0 <= ix < 5 and 0 <= iy < 5
            blocks.append(dict(block_id=f'{ix}_{iy}', cell_x=ix, cell_y=iy,
                               bounds_m=bounds, core=core,
                               split=old_roles[str(iy*5+ix)] if core else None))
    outer = [b for b in blocks if not b['core']]
    order = np.random.default_rng(extension_seed).permutation(len(outer))
    n_holdout = round(.15 * len(outer))
    for i, index in enumerate(order):
        outer[index]['split'] = ('test' if i < n_holdout else
                                  'validation' if i < 2*n_holdout else 'train')
    return blocks


def allocate(blocks, count, buffer_m):
    # Weight clipped edge blocks by their usable rectangular area. Receiver
    # exclusion is applied within each block; quotas are not coverage-weighted.
    areas = np.array([(b['bounds_m'][1]-b['bounds_m'][0]-2*buffer_m) *
                      (b['bounds_m'][3]-b['bounds_m'][2]-2*buffer_m) for b in blocks])
    if count < 1 or np.any(areas <= 0):
        raise ValueError('Invalid count or buffer')
    shares = count * areas / areas.sum()
    quotas = np.floor(shares).astype(int)
    order = np.argsort(-(shares-quotas), kind='stable')
    quotas[order[:count-int(quotas.sum())]] += 1
    return {b['block_id']: int(n) for b, n in zip(blocks, quotas)}


def sample_dataset(width_km, request, source_config, old_roles):
    blocks = make_blocks(width_km, old_roles, request['extension_split_seed'])
    _, inverse = base.projection(source_config['profile'])
    buffer_m, seed = request['boundary_buffer_m'], request['sampling_seed']
    result, quotas = {}, {}
    for split_index, split in enumerate(SPLITS):
        selected = [b for b in blocks if b['split'] == split]
        quotas[split] = allocate(selected, request['sizes'][split], buffer_m)
        scenes = []
        for b in selected:
            ix, iy = b['cell_x'], b['cell_y']
            rng = np.random.default_rng(np.random.SeedSequence([seed, 901, ix+100, iy+100]))
            xmin, xmax, ymin, ymax = b['bounds_m']
            for k in range(quotas[split][b['block_id']]):
                for retries in range(10000):
                    x = float(rng.uniform(xmin+buffer_m, xmax-buffer_m))
                    y = float(rng.uniform(ymin+buffer_m, ymax-buffer_m))
                    lon, lat = inverse.transform(x, y)
                    d = [base.sim.GEOD.inv(lon, lat, r.lon_deg, r.lat_deg)[2]
                         for r in base.ORIGINAL_RXS]
                    if min(d) >= request['min_distance_to_receiver_m']:
                        break
                else:
                    raise ValueError('Receiver exclusion leaves no sampleable area')
                identity = f'{seed}:{ix}:{iy}:{k}'
                sid = 'v6_' + hashlib.sha256(identity.encode()).hexdigest()[:24]
                scenes.append(dict(sample_id=sid, x_m=x, y_m=y, latitude_deg=lat,
                    longitude_deg=lon, cell_x=ix, cell_y=iy, block_id=b['block_id'],
                    split=split, region='core' if b['core'] else 'outer',
                    geometry_replacements=retries,
                    receiver_seeds=[int(s) for s in np.random.SeedSequence(
                        [seed, 902, ix+100, iy+100, k]).generate_state(2)]))
        np.random.default_rng(np.random.SeedSequence([seed, 903, split_index])).shuffle(scenes)
        result[split] = scenes
    return result, blocks, quotas


def verify_geometry(scenes, blocks, request, source_config):
    from scipy.spatial import cKDTree
    block_lookup = {b['block_id']: b for b in blocks}
    all_scenes = [s for split in SPLITS for s in scenes[split]]
    assert len({s['sample_id'] for s in all_scenes}) == len(all_scenes)
    assert len({(s['latitude_deg'], s['longitude_deg']) for s in all_scenes}) == len(all_scenes)
    forward, _ = base.projection(source_config['profile'])
    minimum_rx_distance = math.inf
    for split in SPLITS:
        assert len(scenes[split]) == request['sizes'][split]
        for s in scenes[split]:
            b = block_lookup[s['block_id']]
            assert b['split'] == split == s['split']
            xmin, xmax, ymin, ymax = b['bounds_m']
            margin = min(s['x_m']-xmin, xmax-s['x_m'], s['y_m']-ymin, ymax-s['y_m'])
            assert margin >= request['boundary_buffer_m'] - 1e-6
            assert np.allclose(forward.transform(s['longitude_deg'], s['latitude_deg']),
                               [s['x_m'], s['y_m']], atol=1e-5, rtol=0)
            distances = [base.sim.GEOD.inv(s['longitude_deg'], s['latitude_deg'], r.lon_deg, r.lat_deg)[2]
                         for r in base.ORIGINAL_RXS]
            minimum_rx_distance = min(minimum_rx_distance, *distances)
    assert minimum_rx_distance >= request['min_distance_to_receiver_m']
    xy = {split: np.array([[s['x_m'], s['y_m']] for s in scenes[split]]) for split in SPLITS}
    separation = {}
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        separation[f'{a}_to_{b}'] = float(cKDTree(xy[a]).query(xy[b])[0].min())
        assert separation[f'{a}_to_{b}'] >= 2*request['boundary_buffer_m'] - 1e-6
    return dict(unique_ids_and_positions=True, projection_roundtrip_passed=True,
                geographic_blocks_and_buffers_passed=True, minimum_receiver_distance_m=minimum_rx_distance,
                minimum_cross_split_distance_m=separation)


def protected_files():
    paths = list(ROOT.glob('*.py')) + list(ROOT.glob('*.md'))
    for directory in ['configs', 'docs', 'tests', 'scripts']:
        paths += [p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in ['terrestrial_location_v3', 'terrestrial_location_v4_directional', 'terrestrial_location_v5_lpda92']:
        paths += [p for p in (ROOT/'datasets'/name).rglob('*') if p.is_file()]
        if (ROOT/'datasets'/f'{name}.zip').exists():
            paths.append(ROOT/'datasets'/f'{name}.zip')
    return sorted(set(paths + [base.SIM_SOURCE]))


def replay_checks(freq, models, config, patterns, old_dataset):
    v5.initialize_worker(freq, models, config, patterns)
    checks = []
    for split in SPLITS:
        truth = json.loads((old_dataset/split/'ground_truth.jsonl').open().readline())
        observed = json.loads((old_dataset/split/'inputs.jsonl').open().readline())
        scenario = {k: truth[k] for k in v5.SCENARIO_FIELDS}
        row = v5.simulate_directional(scenario)
        assert {'sample_id': row['sample_id'], **row['input']} == observed
        assert row['truth'] == truth
        checks.append(dict(split=split, sample_id=scenario['sample_id'], exact_v5_replay=True))
    return checks


def quantiles(values):
    return dict(zip(['min','p10','median','p90','max'], map(float, np.percentile(values, [0,10,50,90,100]))))


def summarize(rows):
    result = v5.coverage([r['input'] for r in rows])
    result['regions'] = {}
    for region in ['core', 'outer']:
        subset = [r for r in rows if r['truth']['region'] == region]
        if subset:
            result['regions'][region] = v5.coverage([r['input'] for r in subset])
    result['receivers'] = {}
    for index, name in enumerate(['RX1','RX2']):
        links = [r['truth']['links'][name] for r in rows]
        detected = [10*np.log10(max(i['intensity_value'] for i in r['input']['receivers'][index]['incidents']))
                    for r in rows if r['input']['receivers'][index]['incidents']]
        result['receivers'][name] = dict(
            distance_km=quantiles([l['distance_km'] for l in links]),
            path_loss_800mhz_db=quantiles([l['center_losses_db']['800000000'] for l in links]),
            antenna_gain_800mhz_dbi=quantiles([l['rx_antenna_response']['effective_gain_dbi'] for l in links]),
            strongest_incident_dbm=quantiles(detected) if detected else None)
    return result


def dataset_readme(out, width, request, summary, common):
    lines = [f'# Terrestrial localization v6: {width} x {width} km', '',
        'One sample is one stationary LTE source observed by both receivers for 60 seconds.',
        'This is a new sample of source locations, using the existing v5 LPDA-92 receiver model.', '',
        '## Files', '',
        'Each split has inputs.jsonl, labels.jsonl and ground_truth.jsonl, joined by sample_id.',
        'Use incidents as input and labels for transmitter latitude/longitude. Ground truth is audit-only.',
        'The transmitter type is always terrestrial_lte. Do not use sample IDs or ground truth as features.',
        'Incident lat/lon are RECEIVER coordinates. Intensity is mW; use 10*log10(mW) for dBm.',
        'Keep ongoing incidents with null end/duration fields as censored observations.', '',
        '## Sampling and comparison', '',
        'Same center, sites, fixed antenna pointing, transmit power, frequency, IQ templates, detector and physics as v5.',
        '4 km geographic blocks are anchored to the original 20 km grid; original block roles are retained.',
        'The 40 km square has clipped 2 km strips at its outside edges: 121 rectangles, not 100 shifted blocks.',
        'Exclude 250 m inside every block edge and source positions within 1 km of either receiver.',
        'Within each split, quotas follow usable rectangular block area (before the small receiver exclusion).',
        'Draw uniformly within each block, resampling only geometric exclusions. No detection-based filtering.',
        'Therefore this is buffered, geographically stratified sampling, not uniform over the entire square.',
        'Zero-detection scenes remain in the dataset and still have a source label; they are not source-off negatives.',
        'Report coverage and detected-scene localization error separately; empty inputs cannot give unique positions.',
        'Two datasets share deterministic within-block sample prefixes. A shared ID means identical observations and labels.',
        'common_evaluation.json lists matching core validation/test scenes. No source moves between split roles.',
        'Use shared validation scenes to compare models on the same core task. Reserve test for the final frozen comparison.',
        'Report outer-region metrics separately. The wider set allocates fewer of its 7000 training samples to the core.',
        'These are alternative training datasets: do not concatenate them without deduplicating shared IDs.',
        'A larger region is not evidence of better model accuracy; no model was trained in this generation run.', '',
        '## Coverage', '', '| Split | Samples | Both | RX1 only | RX2 only | Neither | Incidents |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in SPLITS:
        s=summary['splits'][split]
        lines.append(f"| {split} | {s['samples']} | {s['both']} | {s['RX1_only']} | {s['RX2_only']} | {s['neither']} | {s['incidents']} |")
    lines += ['', f"Shared core evaluation: {len(common['validation'])} validation and {len(common['test'])} test samples.", '',
        '## Reproduction', '',
        'From the repository root, with the local IQ/SRTM files and Python dependencies installed:', '',
        '```bash', '../.venv/bin/python generate_location_comparison_v6.py --config configs/location_comparison_v6.json', '```', '',
        'The generator refuses to overwrite either dataset or ZIP. To rerun, copy the config and change both output paths.',
        f"Sampling seed: {request['sampling_seed']}; outer block assignment seed: {request['extension_split_seed']}.",
        'The template schedule keeps the v5 seed 20260921. config.json records effective settings; manifest.json records source/data hashes.',
        'verification.json records geometry, schema, replay and reproducibility checks.', '',
        '## Scope', '',
        'Same full P.452-16/SRTM path calculations as v5; no free-space shortcut or flat-terrain fallback.',
        'The LPDA-92 pattern is a digitized manufacturer-curve approximation, not a calibrated installation measurement.',
        'This fixed-power, fixed-traffic, single-type dataset tests localization under one simulator configuration.', '']
    (out/'README.md').write_text('\n'.join(lines))


def run(request_path):
    started = time.monotonic()
    request = load(request_path)
    if set(request['datasets']) != {'20', '40'} or set(request['sizes']) != set(SPLITS):
        raise ValueError('Specify both widths and all three splits')
    if sum(request['sizes'].values()) != 10000:
        raise ValueError('Each dataset must have 10000 samples')
    outputs = {int(w): (ROOT/p).resolve() for w,p in request['datasets'].items()}
    if len(set(outputs.values())) != 2 or any(p.exists() or p.with_suffix('.zip').exists() for p in outputs.values()):
        raise FileExistsError('Use two distinct, new output paths; existing data are never overwritten')
    source = ROOT/'datasets/terrestrial_location_v3'
    old_v5 = ROOT/'datasets/terrestrial_location_v5_lpda92'
    source_config = load(source/'source_simulation_config.json')
    old_roles = load(source/'split_manifest.json')['block_roles']
    old_v5_config = load(old_v5/'config.json')
    pattern = load(ROOT/'configs/real_antenna_receivers_v5.json')['receiver_pattern']
    patterns = v5.resolve_patterns(source_config, pattern)
    assert patterns == old_v5_config['resolved_patterns']
    config = {**source_config, 'record_format':'v2', 'srtm_dir':str((ROOT/'srtm_data').resolve())}
    assert base.sha256(base.SIM_SOURCE) == source_config['simulator_sha256']
    assert base.sha256(ROOT/'EMILY-X/srsran_loaded.cf32') == source_config['iq_sha256']
    manifest = base.runtime_manifest(config)
    assert manifest['terrain_sha256'] == load(ROOT/'configs/reference_v3_inputs.json')['terrain_sha256']
    manifest['random_streams'] = dict(
        positions='SeedSequence([sampling_seed,901,cell_x+100,cell_y+100])',
        receiver_noise='SeedSequence([sampling_seed,902,cell_x+100,cell_y+100,within_block_index])',
        output_shuffle='SeedSequence([sampling_seed,903,split_index])',
        template_schedule='Unchanged v5 SeedSequence([20260921,40])',
        outer_roles='default_rng(extension_split_seed).permutation(outer_block_count)')
    paths = protected_files()
    before = v5.fingerprint_files(paths)
    manifest.update(protected_files_sha256=before, request_sha256=base.sha256(request_path),
                    generator_sha256=base.sha256(__file__), iq_sha256=source_config['iq_sha256'])
    samples, blocks, quotas, geometry = {}, {}, {}, {}
    for width in outputs:
        samples[width], blocks[width], quotas[width] = sample_dataset(width, request, source_config, old_roles)
        geometry[width] = verify_geometry(samples[width], blocks[width], request, source_config)
    shared = {}
    for split in SPLITS:
        a = {s['sample_id']: s for s in samples[20][split]}
        b = {s['sample_id']: s for s in samples[40][split]}
        shared[split] = sorted(a.keys() & b.keys())
        assert shared[split]
        assert all(a[sid] == b[sid] for sid in shared[split])
    print('Geometry validated. Shared scenes: '+str({k:len(v) for k,v in shared.items()}), flush=True)
    print('Building loaded-IQ templates and replaying three existing v5 scenes...', flush=True)
    freq, models = base.build_models()
    replays = replay_checks(freq, models, config, patterns, old_v5)
    print('Existing v5 inputs and ground truth reproduce exactly.', flush=True)
    summaries, shared_cache = {}, {}
    for width, out in outputs.items():
        run_started = time.monotonic()
        out.mkdir(parents=True)
        resolved = dict(request=request, width_km=width, receiver_pattern=pattern, resolved_patterns=patterns,
            simulation={**config, 'count':10000, 'width_km':width, 'cells':None,
                        'incident_training_target':0, 'sampling':'v6 geographic blocks; all detections retained',
                        'rejection_sampling':'receiver geometry only; no signal-based selection'},
            note='simulation.seed controls the fixed v5 template schedule; request.sampling_seed controls new locations/noise')
        base.write_json(out/'config.json', resolved)
        base.write_json(out/'manifest.json', manifest)
        base.write_json(out/'split_manifest.json', dict(blocks=blocks[width], quotas_per_block=quotas[width],
            sizes=request['sizes'], block_width_m=4000, boundary_buffer_each_side_m=request['boundary_buffer_m'],
            sample_ids={s:[r['sample_id'] for r in samples[width][s]] for s in SPLITS}))
        base.write_json(out/'common_evaluation.json', dict(validation=shared['validation'], test=shared['test'],
            description='Identical core observations/labels in both datasets; never use test for tuning'))
        ordered = [s for split in SPLITS for s in samples[width][split]]
        base.write_jsonl(out/'scenarios.jsonl', ordered)
        (out/'code').mkdir()
        for p in [Path(__file__), Path(v5.__file__), Path(base.__file__), Path(v5.antenna.__file__)]:
            shutil.copyfile(p, out/'code'/p.name)
        shutil.copyfile(ROOT/pattern['data_file'], out/'antenna_pattern.json')
        shutil.copyfile(request_path, out/'requested_config.json')
        rows = {s['sample_id']: shared_cache[s['sample_id']] for s in ordered if s['sample_id'] in shared_cache}
        reused = len(rows)
        pending = [s for s in ordered if s['sample_id'] not in rows]
        try:
            with futures.ProcessPoolExecutor(max_workers=request['workers'], mp_context=multiprocessing.get_context('spawn'),
                    initializer=v5.initialize_worker, initargs=(freq,models,config,patterns)) as pool:
                jobs = {pool.submit(v5.simulate_directional,s):s for s in pending}
                for i, job in enumerate(futures.as_completed(jobs),1):
                    scenario=jobs[job]
                    row=job.result()
                    assert row['sample_id']==scenario['sample_id']
                    assert all(row['truth'][k]==v for k,v in scenario.items())
                    base.validate_row(row)
                    rows[row['sample_id']]=row
                    if i%500==0 or i==len(pending):
                        print(f'{width} km: {i}/{len(pending)} new scenes ({reused} identical scenes reused), {time.monotonic()-run_started:.1f} s',flush=True)
        except Exception as exc:
            base.write_json(out/'FAILED.json',dict(error=repr(exc),completed=len(rows)))
            raise
        # Recompute six rows outside the process pool, checking seed and worker independence.
        v5.initialize_worker(freq,models,config,patterns)
        repeat_ids=[]
        for split in SPLITS:
            for s in [samples[width][split][0],samples[width][split][-1]]:
                assert v5.simulate_directional(s)==rows[s['sample_id']]
                repeat_ids.append(s['sample_id'])
        summary=dict(width_km=width,splits={},new_simulations=len(pending),identical_scenes_reused=reused)
        for split in SPLITS:
            selected=[rows[s['sample_id']] for s in samples[width][split]]
            base.export_rows(out/split,selected)
            summary['splits'][split]=summarize(selected)
        summary['total']=summarize(list(rows.values()))
        summary['wall_seconds_before_packaging']=time.monotonic()-run_started
        summary['shared_scenes']={k:len(v) for k,v in shared.items()}
        base.write_json(out/'summary.json',summary)
        verification=dict(passed=True,geometry=geometry[width],v5_replays=replays,
            fresh_replays_equal_including_ground_truth=repeat_ids,all_rows_validated=True,
            zero_detection_scenes_retained=True,model_training_performed=False,
            table_sha256={f'{s}/{n}':base.sha256(out/s/n) for s in SPLITS for n in TABLES},
            original_files_unchanged=v5.fingerprint_files(paths)==before)
        assert verification['original_files_unchanged']
        base.write_json(out/'verification.json',verification)
        dataset_readme(out,width,request,summary,shared)
        if width==20:
            shared_cache={sid:rows[sid] for split in SPLITS for sid in shared[split]}
        else:
            assert all(rows[sid]==r for sid,r in shared_cache.items())
        summaries[width]=summary
        with zipfile.ZipFile(out.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file(): z.write(p,str(Path(out.name)/p.relative_to(out)))
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            assert z.testzip() is None
        print(f'Completed {out.name}: '+str(summary['total']),flush=True)
    assert v5.fingerprint_files(paths)==before
    print(f'Both datasets finished in {time.monotonic()-started:.1f} seconds.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/location_comparison_v6.json')
    args=p.parse_args()
    run(args.config)


if __name__=='__main__':
    main()
