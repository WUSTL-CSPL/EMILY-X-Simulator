#!/usr/bin/env python3
"""Independently validate the v6 exports and summarize coverage, without training a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import generate_location_comparison_v6 as g
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=g.ROOT


def check_dataset(width):
    root=ROOT/'datasets'/f'terrestrial_location_v6_{width}km'
    manifest=g.load(root/'manifest.json'); verification=g.load(root/'verification.json')
    split_info=g.load(root/'split_manifest.json'); scenarios=g.v5.read_jsonl(root/'scenarios.jsonl')
    proposed={s['sample_id']:s for s in scenarios}
    result={}; coords=set(); ids=set()
    for split in g.SPLITS:
        paths={name:root/split/name for name in g.TABLES}
        tables={name:g.v5.read_jsonl(p) for name,p in paths.items()}
        for name,p in paths.items():
            assert g.base.sha256(p)==verification['table_sha256'][f'{split}/{name}']
        split_ids=[r['sample_id'] for r in tables['inputs.jsonl']]
        assert split_ids==split_info['sample_ids'][split]
        assert len(split_ids)==split_info['sizes'][split]
        assert len(set(split_ids))==len(split_ids) and not ids.intersection(split_ids)
        ids.update(split_ids)
        assert all([r['sample_id'] for r in tables[name]]==split_ids for name in g.TABLES)
        result[split]={}
        for inp,label,truth in zip(*(tables[n] for n in g.TABLES)):
            sid=inp['sample_id']
            xy=(truth['latitude_deg'],truth['longitude_deg'])
            assert xy not in coords
            coords.add(xy)
            assert all(truth[k]==v for k,v in proposed[sid].items())
            row=dict(sample_id=sid,input={'receivers':inp['receivers']},
                     label={k:v for k,v in label.items() if k!='sample_id'},truth=truth)
            g.base.validate_row(row)
            assert label['transmitter_type']=='terrestrial_lte'
            for link in truth['links'].values():
                assert link['propagation_model']=='ITU-R P.452-16 + SRTM'
                assert link['tx_ground_elevation_m']>10 and link['rx_ground_elevation_m']>10
            result[split][sid]=row
    assert len(ids)==10000==len(proposed)
    for p,h in manifest['protected_files_sha256'].items():
        assert g.base.sha256(p)==h, f'Protected file changed: {p}'
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive) as z:
        files=[p for p in root.rglob('*') if p.is_file()]
        assert set(z.namelist())=={str(Path(root.name)/p.relative_to(root)) for p in files}
        for p in files:
            member=str(Path(root.name)/p.relative_to(root))
            assert hashlib.sha256(z.read(member)).hexdigest()==g.base.sha256(p)
    return result,dict(samples=len(ids),tables_verified=9,all_archive_members_match=True,
                       protected_files_unchanged=len(manifest['protected_files_sha256']),
                       archive_bytes=archive.stat().st_size,archive_sha256=g.base.sha256(archive))


def main():
    report_dir=ROOT/'datasets/terrestrial_location_v6_comparison'
    if report_dir.exists(): raise FileExistsError(report_dir)
    datasets={}; checks={}
    for width in (20,40): datasets[width],checks[width]=check_dataset(width)
    common=g.load(ROOT/'datasets/terrestrial_location_v6_20km/common_evaluation.json')
    assert common==g.load(ROOT/'datasets/terrestrial_location_v6_40km/common_evaluation.json')
    overlaps={}
    for split in g.SPLITS:
        a,b=datasets[20][split],datasets[40][split]
        ids=sorted(a.keys() & b.keys())
        assert all(a[sid]==b[sid] for sid in ids)
        overlaps[split]=len(ids)
        if split!='train': assert ids==common[split]
    # Cross-dataset leakage check: train from EITHER set must remain separated
    # from held-out scenes in BOTH sets, including nonshared positions.
    union={split:{} for split in g.SPLITS}
    for ds in datasets.values():
        for split,rows in ds.items(): union[split].update(rows)
    separation={}
    xy={split:[[r['truth']['x_m'],r['truth']['y_m']] for r in rows.values()] for split,rows in union.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(union[a]).intersection(union[b])
        separation[f'{a}_to_{b}']=float(cKDTree(xy[a]).query(xy[b])[0].min())
        assert separation[f'{a}_to_{b}']>=500.-1e-6
    report_dir.mkdir(parents=True)
    g.base.write_json(report_dir/'verification.json',dict(passed=True,datasets=checks,
        identical_shared_rows=overlaps,minimum_cross_dataset_split_distance_m=separation,
        model_training_performed=False))
    summaries={w:g.load(ROOT/'datasets'/f'terrestrial_location_v6_{w}km/summary.json') for w in (20,40)}
    g.base.write_json(report_dir/'comparison.json',summaries)
    config=g.load(ROOT/'datasets/terrestrial_location_v6_20km/config.json')['simulation']
    forward,_=g.base.projection(config['profile'])
    rxxy=[forward.transform(rx.lon_deg,rx.lat_deg) for rx in g.base.ORIGINAL_RXS]
    colors=dict(train='#3478ad',validation='#db9224',test='#6f4fa1')
    fig,axs=plt.subplots(1,2,figsize=(12,5.4),sharex=True,sharey=True)
    for ax,w in zip(axs,(20,40)):
        for split,rows in datasets[w].items():
            pts=g.np.array([[r['truth']['x_m']/1000,r['truth']['y_m']/1000] for r in rows.values()])
            ax.scatter(pts[:,0],pts[:,1],s=3,alpha=.55,c=colors[split],label=split,rasterized=True)
        for i,(x,y) in enumerate(rxxy):
            ax.scatter(x/1000,y/1000,marker='*',s=110,c='#b62c2c',edgecolors='white',linewidths=.6,zorder=5)
            ax.annotate(f'RX{i+1}',(x/1000,y/1000),xytext=(5,5),textcoords='offset points',fontsize=9)
        ax.add_patch(Rectangle((-10,-10),20,20,fill=False,ls='--',lw=1,color='black'))
        ax.set(title=f'{w} x {w} km: 10,000 scenes',xlabel='East of shared center (km)',xlim=(-21,21),ylim=(-21,21))
        ax.set_aspect('equal'); ax.grid(alpha=.15)
    axs[0].set_ylabel('North of shared center (km)')
    axs[0].legend(markerscale=3,loc='lower left')
    fig.suptitle('Geographic splits; dashed box is the original sampling region')
    fig.tight_layout(); fig.savefig(report_dir/'sampling_regions.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.6))
    names=['both','RX1_only','RX2_only','neither']; labels=['Both receivers','RX1 only','RX2 only','Neither']
    palette=['#3e946d','#4a82be','#dcaf4d','#ba5656']; bottom=g.np.zeros(4)
    cohorts=[summaries[20]['total'],summaries[40]['total'],summaries[40]['total']['regions']['core'],summaries[40]['total']['regions']['outer']]
    for name,label,color in zip(names,labels,palette):
        values=g.np.array([100*s[name]/s['samples'] for s in cohorts])
        ax.bar(range(4),values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v>=4: ax.text(i,bottom[i]+v/2,f'{v:.1f}%',ha='center',va='center',fontsize=9)
        bottom+=values
    ax.set_xticks(range(4),['20 km: all','40 km: all','40 km: core','40 km: outer'])
    ax.set(ylabel='Share of proposed scenes (%)',ylim=(0,105),title='Detection coverage (no detection-based filtering)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=4)
    fig.tight_layout(); fig.savefig(report_dir/'detection_coverage.png',dpi=180); plt.close(fig)
    lines=['# Two larger localization datasets (v6)','',
        'We generated two datasets to compare denser sampling with a wider geographic range.',
        'Each has 10,000 scenes: 7,000 training, 1,500 validation, and 1,500 test. A scene is one LTE source and both receivers.',
        'The first covers the original 20 x 20 km square. The second covers a centered 40 x 40 km square.', '',
        '## What stayed the same','',
        'The v5 transmitter, two receiver sites, LPDA-92 antenna curves and fixed pointing, IQ templates, 60-second window, detector, and full P.452/SRTM calculations.',
        'The region center is 39.47016783035665, -114.45433040440209. Receiver antennas are not repointed for the wider region.', '',
        '## Sampling','',
        'We sample new positions directly in geographic train/validation/test blocks. We retain the original 25 core block roles, 4 km block spacing, 250 m inside-edge buffer, and 1 km receiver exclusion.',
        'Keeping the original grid means the 40 km square clips outside blocks into 2 km edge strips: 121 rectangles total. We weight those smaller blocks by usable area.',
        'Sampling is uniform inside each block after geometric exclusions, with fixed per-split quotas. It is not uniform over the unbuffered square.',
        'Outer block roles are assigned before running the simulator. No position is accepted or rejected based on its incidents.',
        'Unlike the historical selected v3/v5 set, empty scenes are retained. Old files were preserved.', '',
        '## Results','', '| Dataset / split | Scenes | Both | RX1 only | RX2 only | Neither | Incidents |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for w in (20,40):
        for split in [*g.SPLITS,'total']:
            s=summaries[w]['total'] if split=='total' else summaries[w]['splits'][split]
            lines.append(f"| {w} km / {split} | {s['samples']} | {s['both']} | {s['RX1_only']} | {s['RX2_only']} | {s['neither']} | {s['incidents']} |")
    lines += ['', '## How to compare models','',
        f"Both datasets contain {overlaps['validation']} identical core validation scenes and {overlaps['test']} identical core test scenes. IDs are listed in each common_evaluation.json.",
        'Train one model per dataset, using the same architecture and tuning budget. Compare on the shared validation scenes first; keep test for the final evaluation.',
        'Report the wider dataset\'s outer-region results separately. Its training set contains 1,734 core scenes and 5,266 outer scenes, so this is a fixed-budget allocation experiment.',
        'A change in overall error combines a different training distribution and a different evaluation domain. It does not by itself show that expanding the region helps the original task.',
        'Use inputs.jsonl as observations and labels.jsonl as targets. Ground truth, block roles, and coordinates derived from labels must not become input features.',
        'Report coverage and localization error on detected scenes separately. Two empty incident lists cannot identify a unique source location; these are not source-absent negatives.',
        'The two training sets share some scenes. They are alternative datasets; do not concatenate without deduplicating sample IDs.', '',
        '## Checks','',
        'All 20,000 exported records passed schema, ID alignment, coordinate, geographic separation, and terrain checks. Every ZIP member matches its dataset file.',
        'Three old v5 scenes reproduced exactly, including observations and ground truth. Six scenes per new dataset were independently recomputed and matched exactly.',
        'Shared rows match in full, including receiver noise and ground truth. Old source and dataset hashes are unchanged. No localization model was trained.', '',
        '## Reproduce','', 'From emily-location-dataset, using the existing parent Python environment:', '',
        '```bash','../.venv/bin/python generate_location_comparison_v6.py --config configs/location_comparison_v6.json',
        '../.venv/bin/python analyze_location_comparison_v6.py','```','',
        'The scripts refuse to overwrite their outputs. For a generation rerun, copy the config and choose new dataset paths; the analysis script above targets the default v6 paths.',
        'Sampling seed: 20260929. Outer split seed: 20260930. Template seed: 20260921, unchanged from v5. Eight worker processes.',
        'Source hashes, runtime versions and terrain hashes are stored in each manifest.json. These archives contain records and code snapshots; large IQ/terrain dependencies remain local.', '',
        '![Sampling regions](sampling_regions.png)', '', '![Detection coverage](detection_coverage.png)', '']
    (report_dir/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(verification=checks,shared=overlaps,coverage={w:summaries[w]['total'] for w in (20,40)}),indent=2))

if __name__=='__main__': main()
