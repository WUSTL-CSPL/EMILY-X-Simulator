#!/usr/bin/env python3
"""Validate all six omnidirectional v8 exports and report their selection bias."""
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import generate_multireceiver_dataset_v8 as g
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT,base,np=g.ROOT,g.base,g.np


def verify_dataset(root,n,width,size):
    cfg=g.load(root/'config.json');verification=g.load(root/'verification.json')
    split_info=g.load(root/'split_manifest.json');all_ids=set();positions=set();result={}
    base.sim.RECEIVERS=[base.sim.Receiver(**r) for r in cfg['receivers']]
    for split in g.SPLITS:
        tables={name:[json.loads(s) for s in (root/split/name).read_text().splitlines()] for name in g.TABLES}
        ids=[r['sample_id'] for r in tables['inputs.jsonl']]
        assert ids==split_info['sample_ids'][split]
        assert len(ids)==cfg['sizes'][split] and len(set(ids))==len(ids) and not all_ids.intersection(ids)
        all_ids.update(ids);result[split]={}
        for name in g.TABLES:
            assert [r['sample_id'] for r in tables[name]]==ids
            assert base.sha256(root/split/name)==verification['table_sha256'][f'{split}/{name}']
        for inp,label,truth in zip(*(tables[name] for name in g.TABLES)):
            row=dict(sample_id=inp['sample_id'],input={'receivers':inp['receivers']},
                     label={k:v for k,v in label.items() if k!='sample_id'},truth=truth)
            base.validate_row(row)
            g.validate_randomization(row,n,cfg['request'])
            assert len(inp['receivers'])==n and g.all_detected(row,n)
            assert label['transmitter_type']=='terrestrial_lte'
            assert truth['receivers']==cfg['receivers'] and len(truth['receiver_seeds'])==n
            assert set(truth['links'])=={r['name'] for r in cfg['receivers']}
            assert 'receiver_antenna_patterns' not in truth
            for r in cfg['receivers']:
                assert r['antenna_gain_dbi']==0. and r['height_agl_m']==2. and r['noise_figure_db']==6.
                assert 'rx_antenna_response' not in truth['links'][r['name']]
            point=(label['latitude_deg'],label['longitude_deg']);assert point not in positions;positions.add(point)
            assert abs(truth['x_m'])<width*500 and abs(truth['y_m'])<width*500
            assert truth['split']==split
            result[split][row['sample_id']]=row
    assert len(all_ids)==size
    with zipfile.ZipFile(root.with_suffix('.zip')) as z:
        files=[p for p in root.rglob('*') if p.is_file()]
        assert set(z.namelist())=={str(Path(root.name)/p.relative_to(root)) for p in files}
        for p in files:
            assert hashlib.sha256(z.read(str(Path(root.name)/p.relative_to(root)))).hexdigest()==base.sha256(p)
    return result


def describe(values):
    x=np.asarray(values,dtype=float)
    return dict(count=len(x),min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()),
                quantiles=dict(zip(['p05','p25','p50','p75','p95'],map(float,np.quantile(x,[.05,.25,.5,.75,.95])))))


def randomization_report(datasets,request,campaign,out):
    report={'proposal_domains':{},'exports':{}};domain_rows={}
    bins=np.linspace(33,43,11)
    for width in (20,40):
        records=[]
        for split in g.SPLITS:
            with (campaign/f'proposals_{width}km_{split}.csv').open() as handle:
                records.extend(dict(row,split=split) for row in csv.DictReader(handle))
        domain_rows[width]=records
        powers=np.array([float(r['transmitter_power_dbm']) for r in records])
        offsets=np.array([[float(r[f'measurement_offset_RX{i+1}_db']) for i in range(4)] for r in records])
        domain={'power_dbm':describe(powers),
                'offsets_db':{f'RX{i+1}':describe(offsets[:,i]) for i in range(4)},
                'correlation_order':['power_dbm','RX1_offset','RX2_offset','RX3_offset','RX4_offset'],
                'correlation':np.corrcoef(np.column_stack([powers,offsets]).T).tolist(),
                'acceptance_by_power_bin':[]}
        for lower,upper in zip(bins[:-1],bins[1:]):
            mask=(powers>=lower)&(powers<upper)
            subset=[r for r,yes in zip(records,mask) if yes]
            domain['acceptance_by_power_bin'].append(dict(lower_dbm=float(lower),upper_dbm=float(upper),proposed=len(subset),
                eligible_3rx=sum(int(r['eligible_3rx']) for r in subset),eligible_4rx=sum(int(r['eligible_4rx']) for r in subset)))
        report['proposal_domains'][str(width)]=domain
    for key,ds in datasets.items():
        rows=[r for split in g.SPLITS for r in ds[split].values()];n=len(rows[0]['input']['receivers'])
        powers=[r['truth']['transmitter_power_dbm'] for r in rows]
        offsets=np.array([[r['truth']['receiver_measurement_offsets_db'][f'RX{i+1}'] for i in range(n)] for r in rows])
        report['exports'][key]=dict(power_dbm=describe(powers),
            per_split_power_dbm={split:describe([r['truth']['transmitter_power_dbm'] for r in ds[split].values()]) for split in g.SPLITS},
            offsets_db={f'RX{i+1}':describe(offsets[:,i]) for i in range(n)})
    base.write_json(out/'randomization_statistics.json',report)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,width in zip(axes,(20,40)):
        rows=domain_rows[width]
        ax.hist([float(r['transmitter_power_dbm']) for r in rows],bins=bins,density=True,histtype='step',lw=2,label='All proposals')
        for n in (3,4):
            ds=datasets[f'{n}rx_{width}km_10000']
            ax.hist([r['truth']['transmitter_power_dbm'] for split in g.SPLITS for r in ds[split].values()],bins=bins,density=True,histtype='step',lw=1.6,label=f'Retained: {n} RX')
        ax.set(title=f'{width} x {width} km',xlabel='TX conducted power (dBm)',ylabel='Probability density per dB')
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Power distribution before and after all-receiver selection');fig.tight_layout()
    fig.savefig(out/'power_selection.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));rows=domain_rows[40]
    for i in range(4):
        ax.hist([float(r[f'measurement_offset_RX{i+1}_db']) for r in rows],bins=np.linspace(-9,9,46),density=True,histtype='step',label=f'RX{i+1}',lw=1.2)
    x=np.linspace(-9,9,200);ax.plot(x,np.exp(-x*x/8)/(2*np.sqrt(2*np.pi)),ls='--',color='black',label='N(0, 2 squared)')
    ax.set(xlabel='Readout offset (dB)',ylabel='Probability density',title='Independent per-scene receiver measurement offsets (40 km proposals)');ax.legend()
    fig.tight_layout();fig.savefig(out/'measurement_offsets.png',dpi=160);plt.close(fig)
    lines=['','## Power and measurement error after selection','',
           '| Dataset | Mean TX dBm | TX standard deviation (dB) | RX offset standard deviations (dB) |',
           '| --- | ---: | ---: | --- |']
    for key,r in report['exports'].items():
        stds=', '.join(f"{name}: {v['std']:.2f}" for name,v in r['offsets_db'].items())
        lines.append(f"| {key} | {r['power_dbm']['mean']:.2f} | {r['power_dbm']['std']:.2f} | {stds} |")
    lines+=['','Proposed TX powers have theoretical mean 38 dBm and standard deviation 2.89 dB. Proposed receiver offsets have mean 0 and standard deviation 2 dB.',
            'Power is drawn independently of location before filtering. Requiring detection at every receiver can couple location and power in the retained data. The CSV audits and randomization_statistics.json quantify this.',
            'Both changes were added before EMILY processing. Extra audit fields are only in ground_truth.jsonl: transmitter_power_dbm, receiver_measurement_offsets_db, measurement_offset_sigma_db and randomization_seed. Per-link physical powers/path loss remain unshifted; measured_noise_floor_per_rbw_dbm includes the readout offset.',
            'These offsets are independent across receivers and scenes, but constant across frequency/time within a scene. They are Gaussian in dB, hence multiplicative in mW. Their 2 dB scale is a chosen stress-test level, not fitted to site measurements.',
            'The added unknown TX power and receiver errors may make localization harder. Comparing receiver intensities in dB (for corresponding spectral products) can help remove common TX-power changes, but does not remove independent receiver errors or terrain effects.',
            'The fixed waveform/traffic and a single random power draw per location still limit realism. These datasets alone do not demonstrate generalization to real measurements.',
            '', '![Power selection](power_selection.png)', '', '![Measurement offsets](measurement_offsets.png)','']
    return lines


def main():
    request=g.load(ROOT/'configs/multireceiver_v8.json')
    campaign=ROOT/request['campaign_dir'];completion=g.load(campaign/'completion.json')
    timing_note='Final generation invocation after a restart; earlier work reused from cache' if (campaign/'resume_context.json').exists() else 'Generation time'
    out=ROOT/'datasets/terrestrial_location_v8_comparison'
    if out.exists():raise FileExistsError(out)
    datasets={};checks={};summaries={};roots={}
    for n in (3,4):
        for width,size in [(20,1000),(20,10000),(40,10000)]:
            key=f'{n}rx_{width}km_{size}';root=ROOT/f"{request['output_prefix']}_{key}"
            datasets[key]=verify_dataset(root,n,width,size);roots[key]=root
            summaries[key]=g.load(root/'summary.json')
            checks[key]=dict(samples=size,all_receivers_have_incidents=True,omnidirectional_only=True,
                nine_tables_aligned_and_hashed=True,zip_members_match=True,zip_bytes=root.with_suffix('.zip').stat().st_size)
            print(f'Verified {key}',flush=True)
    for n in (3,4):
        a=datasets[f'{n}rx_20km_1000'];b=datasets[f'{n}rx_20km_10000']
        for split in g.SPLITS:
            assert set(a[split])<=set(b[split])
            assert all(row==b[split][sid] for sid,row in a[split].items())
    union={s:{} for s in g.SPLITS};by_id={};owners={};common_counts={}
    for key,ds in datasets.items():
        for split,rows in ds.items():
            for sid,row in rows.items():
                t=row['truth'];p=(t['x_m'],t['y_m'])
                assert owners.setdefault(t['block_id'],split)==split
                if sid in by_id:
                    old=by_id[sid]
                    assert old['label']==row['label'] and old['truth']['split']==split
                    assert old['truth']['transmitter_power_dbm']==t['transmitter_power_dbm']
                    assert all(old['truth']['receiver_measurement_offsets_db'][f'RX{i+1}']==t['receiver_measurement_offsets_db'][f'RX{i+1}'] for i in range(3))
                    assert old['input']['receivers'][:3]==row['input']['receivers'][:3]
                    assert {k:old['truth']['links'][k] for k in ['RX1','RX2','RX3']}=={k:t['links'][k] for k in ['RX1','RX2','RX3']}
                by_id[sid]=row;union[split][sid]=p
        common=g.load(roots[key]/'common_validation_test.json')
        common_counts[key]={}
        for other,other_ds in datasets.items():
            if key==other:continue
            expected={s:sorted(set(ds[s])&set(other_ds[s])) for s in ['validation','test']}
            assert common[roots[other].name]==expected
            common_counts[key][other]={s:len(ids) for s,ids in expected.items()}
    distances={}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(union[a]).intersection(union[b])
        d=float(cKDTree(list(union[a].values())).query(list(union[b].values()))[0].min())
        assert d>=500.-1e-6;distances[f'{a}_to_{b}']=d
    protected=g.load(campaign/'protected_files.json')
    assert all(base.sha256(p)==h for p,h in protected.items())
    pilot_ids=set();pilot_positions=set()
    for width in (20,40):
        for line in (ROOT/f'runs/multireceiver_v8_pilot/{width}km_rows.jsonl').read_text().splitlines():
            row=json.loads(line);pilot_ids.add(row['sample_id'])
            pilot_positions.add((row['label']['latitude_deg'],row['label']['longitude_deg']))
    assert not pilot_ids.intersection(by_id)
    assert not pilot_positions.intersection((r['label']['latitude_deg'],r['label']['longitude_deg']) for r in by_id.values())
    out.mkdir(parents=True)
    base.write_json(out/'verification.json',dict(passed=True,datasets=checks,nested_small_datasets=True,
        shared_first_three_receivers_identical=True,minimum_cross_dataset_split_distance_m=distances,
        original_files_unchanged=len(protected),all_six_sample_count=sum(c['samples'] for c in checks.values()),
        unique_exported_source_positions=len(by_id),model_training_performed=False))
    base.write_json(out/'common_evaluation_counts.json',common_counts)
    base.write_json(out/'summary.json',dict(datasets=summaries,generation=completion))
    config=g.load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    receivers=g.receivers_for(request,config);forward,_=base.projection(config['profile'])
    distances_rx={f'{a.name}-{b.name}':base.sim.GEOD.inv(a.lon_deg,a.lat_deg,b.lon_deg,b.lat_deg)[2]/1000
        for i,a in enumerate(receivers) for b in receivers[i+1:]}
    base.write_json(out/'receivers.json',dict(receivers=[asdict(r) for r in receivers],pairwise_distance_km=distances_rx))
    colors=dict(train='#327bac',validation='#d99a24',test='#7552a5')
    fig,axs=plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True)
    for ri,n in enumerate((3,4)):
        for ci,(width,size) in enumerate([(20,1000),(20,10000),(40,10000)]):
            ax=axs[ri,ci];ds=datasets[f'{n}rx_{width}km_{size}']
            for split,rows in ds.items():
                xy=np.array([[r['truth']['x_m']/1000,r['truth']['y_m']/1000] for r in rows.values()])
                ax.scatter(xy[:,0],xy[:,1],s=3 if size==1000 else 1.5,alpha=.5,c=colors[split],label=split,rasterized=True)
            for r in receivers[:n]:
                x,y=forward.transform(r.lon_deg,r.lat_deg)
                ax.scatter(x/1000,y/1000,marker='*',s=95,c='#c02e2e',edgecolors='white',linewidths=.5,zorder=5)
                ax.annotate(r.name,(x/1000,y/1000),xytext=(3,4),textcoords='offset points',fontsize=8)
            ax.add_patch(Rectangle((-width/2,-width/2),width,width,fill=False,ls='--',lw=.7,color='black'))
            ax.set(title=f'{n} receivers / {width} km / {size:,} scenes',xlim=(-21,21),ylim=(-21,21),aspect='equal')
            ax.grid(alpha=.12)
            if ri==1:ax.set_xlabel('East of shared center (km)')
            if ci==0:ax.set_ylabel('North of shared center (km)')
    axs[0,0].legend(loc='lower left',fontsize=8,markerscale=3)
    fig.suptitle('Retained positions: every included receiver has an incident')
    fig.tight_layout();fig.savefig(out/'retained_positions.png',dpi=160);plt.close(fig)
    # Acceptance chart describes proposed scenes, not the conditioned exports (which are 100%).
    fig,ax=plt.subplots(figsize=(8,4.8));labels=[];values=[]
    for width in (20,40):
        reports=completion['proposal_reports'][str(width)]
        count=sum(r['proposals'] for r in reports.values())
        for n in (3,4):
            values.append(100*sum(r[f'eligible_{n}rx'] for r in reports.values())/count)
            labels.append(f'{n} receivers\n{width} x {width} km')
    bars=ax.bar(labels,values,color=['#4d8faf','#76a97b','#4d8faf','#76a97b'])
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=3)
    ax.set(ylabel='Proposals detected by all included receivers (%)',ylim=(0,110),title='Acceptance before filtering; all exported scenes pass')
    fig.tight_layout();fig.savefig(out/'proposal_acceptance.png',dpi=160);plt.close(fig)
    lines=['# Six multi-receiver datasets (v8)','',
        'These datasets use the original omnidirectional receiver model. Every retained scene contains at least one incident at every included receiver.',
        'One scene is one LTE source observed for 60 seconds. TX conducted power is sampled uniformly in dBm from 33 to 43 and is shared by all receivers for that scene.',
        'Each receiver gets an independent Gaussian readout offset with mean 0 and standard deviation 2 dB, fixed within the scene. It shifts the full received spectrum before detection; original thermal noise and slow fading remain.',
        'The added offset models measurement error, not additional time-varying fading. Because the detector uses relative contrast, the offset mainly changes intensity rather than detection. Lower TX power changes signal-to-noise ratio and detection.',
        'Carrier, waveform, transmitter antenna and receiver hardware remain unchanged. TX output scales every emission component; nonlinear PA behavior is not recomputed at each power.', '',
        '## Downloads and sizes','',
        '| Dataset | Train | Validation | Test | Incident records | ZIP |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for key in datasets:
        s=summaries[key]['splits'];inc=sum(x['incidents'] for x in s.values());root=roots[key]
        lines.append(f"| {key} | {s['train']['samples']} | {s['validation']['samples']} | {s['test']['samples']} | {inc:,} | [Download](../{root.name}.zip) |")
    lines+=['','## Receiver sites','','| Site | Latitude | Longitude |','| --- | ---: | ---: |']
    for r in receivers:lines.append(f'| {r.name} | {r.lat_deg:.8f} | {r.lon_deg:.8f} |')
    lines+=['', 'RX1 and RX2 are unchanged. RX3 is 5 km west and 2 km south of the shared sampling center; RX4 is 4 km east and 6 km south.',
        f"Pairwise site distances range from {min(distances_rx.values()):.2f} to {max(distances_rx.values()):.2f} km. The sites form a non-collinear layout.",
        'All sites have 0 dBi omnidirectional receive gain, 2 m antenna height, 6 dB noise figure and no extra path loss. The original directional transmitter sector remains unchanged.',
        'A separate 200-scene pilot per width uses only training blocks. Its positions are excluded from these exports; see runs/multireceiver_v8_pilot/summary.json for results.', '',
        '## What filtering changes','',
        'We draw proposals in fixed geographic split blocks, proportional to usable block area, and then retain the first eligible N per split. Eligibility means every included receiver has a nonempty incident list.',
        'This is a dataset of jointly detectable sources. Filtering can favor stronger sources as well as easier locations; the retained power distribution need not be uniform. The 40 km square includes areas that contribute few or no retained samples.',
        'We keep the original grid roles, 250 m inside-edge buffers, and a 1 km exclusion around all four candidate sites. The 40 km grid has clipped edge blocks so the original held-out regions do not move.',
        'Full proposal positions, masks and rejections remain in the local campaign CSVs and SQLite cache; each archive includes a selection report.', '',
        '| Proposal domain | Evaluated proposals | Eligible at 3 RX | Eligible at 4 RX |',
        '| --- | ---: | ---: | ---: |']
    for width in (20,40):
        r=completion['proposal_reports'][str(width)];total=sum(s['proposals'] for s in r.values())
        three=sum(s['eligible_3rx'] for s in r.values());four=sum(s['eligible_4rx'] for s in r.values())
        lines.append(f'| {width} x {width} km | {total:,} | {three:,} ({three/total:.2%}) | {four:,} ({four/total:.2%}) |')
    lines+=['', 'Counts include the final full proposal batch, so eligible counts may exceed the export quota. Shared source positions are simulated once and reused exactly.', '',
        '## Training and comparisons','',
        'Use inputs.jsonl as observations and labels.jsonl as source targets. Incident latitude/longitude are receiver positions. Convert intensity from mW to dBm if appropriate; preserve censored null duration/end fields.',
        'Do not use ground-truth geometry, path loss, source parameters or sample IDs as features. The transmitter class is always terrestrial_lte.',
        'This is multi-receiver localization from signal observations. Incidents do not supply arrival angles or precise arrival times, so it is not a direct angle-based triangulation dataset.',
        'Small datasets are subsets of the corresponding large datasets. Use shared validation/test IDs when comparing training sizes.',
        'Three- and four-receiver datasets have different detection conditions and therefore somewhat different source distributions. common_validation_test.json lists matched positions for controlled evaluation.',
        'Do not concatenate these datasets as independent data. Shared IDs keep the same geographic split across all six versions. Fit preprocessing on training only and reserve test for the final evaluation.',
        'All-receiver detection makes complete-input training possible, but does not guarantee that location is uniquely recoverable from the incidents.', '',
        '## Reproduce','', '```bash',
        '../.venv/bin/python generate_multireceiver_dataset_v8.py --config configs/multireceiver_v8.json',
        '../.venv/bin/python analyze_multireceiver_v8.py','```','',
        'Add --resume to generation after an interrupted run. Resume verifies code, configuration, runtime, IQ and terrain before reusing the cache. To create a different campaign, copy the config and change output_prefix and campaign_dir.',
        'The analysis command targets this default campaign and refuses to overwrite its report directory.',
        'Reproduction also requires the existing v3 reference dataset for configuration, block roles and replay checks, plus the local IQ/SRTM dependencies. The exported ZIP is not a standalone simulator installation.', '',
        '## Verification','',
        'All 42,000 exported samples passed receiver-count, detection, schema, coordinate and ID checks. Archive members match exported files. Small datasets are exact subsets; shared first-three-receiver observations match.',
        'Three old two-receiver scenes reproduced exactly after extending to four receivers. Six scenes from each exported dataset were recomputed exactly. All original protected files remained unchanged.',
        f"{timing_note}: {completion['elapsed_s']/60:.1f} minutes. Unique full four-receiver simulations: {completion['unique_simulated_scenes']:,}. No model training was performed.", '',
        '![Retained positions](retained_positions.png)', '', '![Proposal acceptance](proposal_acceptance.png)', '']
    lines+=randomization_report(datasets,request,campaign,out)
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(checks=checks,minimum_cross_split_distance_m=distances),indent=2),flush=True)

if __name__=='__main__':main()
