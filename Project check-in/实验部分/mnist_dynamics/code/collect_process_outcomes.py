"""Preserve direct, log-derived and unavailable historical process exit evidence."""
import json
import re
from pathlib import Path
from project_scope import load_scope

ROOT=Path(__file__).resolve().parents[1]


def main():
    assert json.loads((ROOT/'results/qa.json').read_text())['complete']
    assert json.loads((ROOT/'metadata/checkpoint_audit.json').read_text())['passed']
    workspace=next(p for p in ROOT.parents if (p/'log.md').is_file())
    successful={}
    pattern=re.compile(r'^- \d{4}-\d\d-\d\dT\S+：完成 `([^`]+)`，100000 steps，')
    for line in (workspace/'log.md').read_text(encoding='utf-8-sig').splitlines():
        match=pattern.match(line)
        if match:successful[match.group(1)]=line
    rows=[]
    manifest,scope,_,_=load_scope()
    for entry in manifest:
        folder=ROOT/entry['output']
        assert json.loads((folder/'status.json').read_text())['state']=='complete'
        exit_file=folder/'process_exit.json'
        if exit_file.exists():
            saved=json.loads(exit_file.read_text());assert saved['exit_code']==0
            row=dict(run_id=entry['run_id'],exit_code=0,source='direct_process_exit_file',
                     evidence=exit_file.relative_to(ROOT).as_posix())
        elif entry['run_id'] in successful:
            row=dict(run_id=entry['run_id'],exit_code=0,source='derived_from_scheduler_success_log',
                     evidence=successful[entry['run_id']],
                     rationale='The scheduler writes this completion line only after process.poll() returns 0.')
        else:
            row=dict(run_id=entry['run_id'],exit_code=None,source='not_retained',
                     evidence='Complete output artifacts passed independent metrics and checkpoint audits; the original exit code was not retained.')
        rows.append(row)
    result=dict(runs=len(rows),artifacts_complete=True,all_known_exit_codes_zero=True,
                direct_codes=sum(r['source']=='direct_process_exit_file' for r in rows),
                derived_codes=sum(r['source']=='derived_from_scheduler_success_log' for r in rows),
                not_retained=sum(r['source']=='not_retained' for r in rows),details=rows)
    (ROOT/'metadata/process_outcomes.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='details'}))


if __name__=='__main__':main()
