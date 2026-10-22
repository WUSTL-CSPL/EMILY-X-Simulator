#!/usr/bin/env python3
"""Restore small ignored v8 metadata from tracked reference files.

IQ and terrain remain external. Existing metadata must match the required
reference fields; no existing file is overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def same_fields(actual,expected):
    if isinstance(expected,dict):
        return isinstance(actual,dict) and all(k in actual and same_fields(actual[k],v) for k,v in expected.items())
    return actual==expected


def prepare(root=ROOT):
    root=Path(root)
    jobs=[('source_simulation_config.json','datasets/terrestrial_location_v3/source_simulation_config.json'),
          ('split_manifest.json','datasets/terrestrial_location_v3/split_manifest.json'),
          ('environment.json','runs/multireceiver_v8/manifest.json')]
    for name,destination in jobs:
        expected=json.loads((root/'reference/v8'/name).read_text())
        if name=='environment.json':expected.pop('snapshot_kind',None)
        if name=='source_simulation_config.json':expected['srtm_dir']=str(root/'srtm_data')
        path=root/destination
        if path.exists():
            actual=json.loads(path.read_text())
            checked=dict(expected)
            if name=='source_simulation_config.json':
                checked.pop('srtm_dir')
                if Path(actual['srtm_dir']).resolve()!=(root/'srtm_data').resolve():
                    raise ValueError(f'Existing terrain path differs: {path}; inspect before reusing this checkout')
            if not same_fields(actual,checked):raise ValueError(f'Existing metadata differs; left untouched: {path}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive creation avoids replacing existing metadata.
            with path.open('x') as handle:json.dump(expected,handle,indent=2);handle.write('\n')
    return [str(root/target) for _,target in jobs]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    for path in prepare():print(path)


if __name__=='__main__':main()
