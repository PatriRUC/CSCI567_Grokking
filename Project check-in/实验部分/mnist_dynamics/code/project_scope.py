"""Resolve the original plan and the user's explicit cancellation revision."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def group_key(config):
    return config['phase'],int(config['n']),float(config['alpha'])


def load_scope():
    current=json.loads((ROOT/'configs/manifest.json').read_text(encoding='utf-8-sig'))
    original_file=ROOT/'configs/manifest_original_42.json'
    original=json.loads(original_file.read_text(encoding='utf-8-sig')) if original_file.exists() else current
    assert len(original)==42 and len({e['run_id'] for e in original})==42
    cancellation_file=ROOT/'metadata/user_cancellation.json'
    cancellation=json.loads(cancellation_file.read_text(encoding='utf-8-sig')) if cancellation_file.exists() else {}
    if cancellation.get('original_manifest_sha256'):
        assert hashlib.sha256(original_file.read_bytes()).hexdigest()==cancellation['original_manifest_sha256']
    cancelled={e['run_id'] for e in cancellation.get('cancelled_runs',[])}
    original_by_id={e['run_id']:e for e in original}
    assert cancelled<=original_by_id.keys()
    expected=[e for e in original if e['run_id'] not in cancelled]
    effective=[e for e in current if e['run_id'] not in cancelled]
    assert len({e['run_id'] for e in effective})==len(effective)
    assert {e['run_id'] for e in effective}=={e['run_id'] for e in expected}
    assert all(e==original_by_id[e['run_id']] for e in effective)
    configs={e['run_id']:json.loads((ROOT/e['config']).read_text()) for e in effective}
    counts=Counter(group_key(c) for c in configs.values())
    roles={run_id:('primary' if counts[group_key(c)]==3 else 'supplementary_single_seed') for run_id,c in configs.items()}
    assert all(count in (1,3) for count in counts.values())
    scope=dict(original_expected_runs=len(original),expected_runs=len(expected),cancelled_by_user=len(cancelled),
               cancelled_run_ids=sorted(cancelled),scope_change_request=cancellation.get('request'),
               primary_runs=sum(role=='primary' for role in roles.values()),
               supplementary_runs=sum(role!='primary' for role in roles.values()),
               primary_fig3e_ns=sorted({c['n'] for run_id,c in configs.items() if c['phase']=='fig3e' and roles[run_id]=='primary'}),
               completion_definition='All user-retained runs, excluding explicitly cancelled jobs; supplementary single seeds do not receive three-seed means.')
    return effective,scope,counts,roles
