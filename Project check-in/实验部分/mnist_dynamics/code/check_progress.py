"""Read local status files without importing the training runtime."""
import json
from pathlib import Path
from project_scope import load_scope

ROOT=Path(__file__).resolve().parents[1]
manifest,scope,_,_=load_scope()
complete=[];active=[];failed=[]
for entry in manifest:
    file=ROOT/entry['output']/'status.json'
    if not file.exists():continue
    status=json.loads(file.read_text())
    row=dict(run=entry['run_id'],**status)
    if status['state']=='complete':complete.append(entry['run_id'])
    elif status['state']=='failed':failed.append(row)
    else:active.append({k:row[k] for k in ('run','state','step','elapsed_seconds') if k in row})
print(json.dumps(dict(completed=len(complete),total=len(manifest),cancelled_by_user=scope['cancelled_by_user'],active=active,failed=failed,
                     last_complete=complete[-3:]),indent=2))
