#!/usr/bin/env python3
"""Portable entry point for original v8 terrestrial generation with 30-second windows.

Source snapshots are preserved byte-for-byte. Paths and the current Python build
are recorded for the new machine; IQ, terrain, simulator, package versions and
split geometry remain checked against the shipped reference.
"""
import argparse
import importlib.metadata
import sys
import generate_multireceiver_v8_30s_million as legacy
import generate_multireceiver_v8_30s_million_profiled as profiled
from generate_multireceiver_v8_30s_million import *

advance=profiled.advance

def portable_runtime(config):
    provenance=g.load(ROOT/'PROVENANCE.json')
    return dict(python=sys.version,
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','pyproj','pycraf','astropy')},
        simulator_git_commit=provenance['upstream_commit'],simulator_source='vendored snapshot; no Git checkout required',
        simulator_sha256=base.sha256(base.SIM_SOURCE),
        terrain_sha256={str(f.relative_to(config['srtm_dir'])):base.sha256(f) for f in sorted(Path(config['srtm_dir']).rglob('*.hgt'))},
        platform=sys.platform,source_machine_python=provenance['source_machine_python'])

def run(args):
    request=g.load(args.config);campaign=ROOT/request['campaign_dir']
    assert request['duration_s']==30
    assert request['reuse_v8_cache'] is False, '60-second cache cannot be reused for 30-second observations'
    assert request['only_small'] is True, 'Only the two 20 km datasets are authorized'
    assert request['small_size']==request['large_size']
    assert request['small_splits']==request['large_splits']
    assert sum(request['small_splits'].values())==request['small_size']
    assert sum(request['large_splits'].values())==request['large_size']
    assert all(0<request['small_splits'][s]<=request['large_splits'][s] for s in g.SPLITS)
    assert request['batch_size']>0 and request['workers']>0 and request['shard_samples']>0
    if campaign.exists() and not args.resume:raise FileExistsError('Use --resume for an existing campaign')
    campaign.mkdir(parents=True,exist_ok=True)
    lock=(campaign/'campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=g.load(ROOT/'datasets/terrestrial_location_v3/source_simulation_config.json')
    config={**config,'duration_s':request['duration_s'],'srtm_dir':str(ROOT/'srtm_data')}
    source_manifest=g.load(ROOT/'runs/multireceiver_v8/manifest.json')
    assert base.sha256(base.__file__)==config['generator_sha256']
    assert base.sha256(base.SIM_SOURCE)==config['simulator_sha256']
    assert base.sha256(ROOT/'EMILY-X/srsran_loaded.cf32')==config['iq_sha256']
    assert base.sha256(g.__file__)==source_manifest['signature_data']['code_sha256'][Path(g.__file__).name]
    runtime=portable_runtime(config)
    runtime['scope']='Original v8 physical chain with 30-second observations, random 33-43 dBm TX power and per-receiver Gaussian measurement offsets.'
    runtime['random_streams']={'sampling_seed':request['sampling_seed'],
        'implementation':'generate_multireceiver_dataset_v8.Proposals and randomization',
        'template_schedule':'SeedSequence([source_config.seed,40])',
        'duration':'All signals regenerated natively at 30 seconds; no old scene cache reuse.'}
    for field in ('packages','terrain_sha256','simulator_git_commit'):
        assert runtime[field]==source_manifest[field],f'Changed v8 dependency: {field}'
    sources=[Path(__file__),Path(g.__file__),Path(base.__file__),base.SIM_SOURCE,Path(legacy.__file__),Path(profiled.__file__)]
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
            # One shared four-receiver simulation supplies independently filtered 3/4-RX datasets.
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
    parser.add_argument('--config',type=Path,default=ROOT/'configs/portable_generate_million.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-batches',type=int,help='Checkpoint after this many batches; useful for restart checks.')
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,legacy.stopping);signal.signal(signal.SIGINT,legacy.stopping)
    from scripts.prepare_v8_inputs import prepare
    prepare(ROOT)
    g.initialize_worker=profiled.initialize_profiled
    run(args)


if __name__=='__main__':main()
