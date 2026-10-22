#!/usr/bin/env python3
"""Read progress without importing the simulator or loading samples into memory."""
import json
import os
from pathlib import Path
import sqlite3
import time

root=Path(__file__).resolve().parents[1]
campaign=root/'runs/multireceiver_v8_30s_million'
if not (campaign/'status.json').exists():
    raise SystemExit('Campaign is not initialized yet. Check runs/multireceiver_v8_30s_million.log.')
status=json.loads((campaign/'status.json').read_text())
request=json.loads((root/'configs/multireceiver_v8_30s_million.json').read_text())
process=json.loads((campaign/'process.json').read_text())
try:
    os.kill(process['pid'],0)
    process_alive=True
except ProcessLookupError:
    process_alive=False
with sqlite3.connect('file:'+str(campaign/'scene_cache.sqlite')+'?mode=ro',uri=True) as db:
    states={split:json.loads(state) for split,state in db.execute('select split,state from pools where width=20')}
counts={str(n):{split:min(request['small_splits'][split],states.get(split,{}).get('counts',{}).get(str(n),0)) for split in request['small_splits']} for n in (3,4)}
proposed=sum(s['proposed'] for s in states.values())
seconds=sum(s['simulation_seconds'] for s in states.values())
remaining=0
for split,target in request['small_splits'].items():
    state=states.get(split)
    acceptance=state['counts']['4']/state['proposed'] if state and state['proposed'] else .94
    remaining+=max(0,target-counts['4'][split])/max(acceptance,.01)
report=dict(status=status,process_alive=process_alive,pid=process['pid'],accepted_samples=counts,
    total_accepted={n:sum(v.values()) for n,v in counts.items()},candidate_scenes=proposed,
    elapsed_hours=(time.time()-process['started_unix'])/3600,
    committed_batch_candidates_per_second=proposed/seconds if seconds else None,
    approximate_remaining_simulation_hours=remaining*seconds/proposed/3600 if proposed else None,
    estimate_note='Rough estimate from committed batches; excludes export and validation; acceptance estimates can change.',
    datasets={str(n):dict(path=str(root/f"{request['output_prefix']}_{n}rx_20km_1000000"),complete=(root/f"{request['output_prefix']}_{n}rx_20km_1000000/COMPLETE.json").exists()) for n in (3,4)})
print(json.dumps(report,indent=2))
