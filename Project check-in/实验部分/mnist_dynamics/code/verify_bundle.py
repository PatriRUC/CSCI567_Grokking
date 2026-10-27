"""Verify the review ZIP contents offline using only Python's standard library."""
import argparse
import csv
import hashlib
import io
import json
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle')
    a=p.parse_args()
    with zipfile.ZipFile(a.bundle) as z:
        assert z.testzip() is None, 'ZIP CRC failed'
        hashes=json.loads(z.read('mnist_dynamics/manifest_sha256.json'))
        for path,expected in hashes.items():
            actual=hashlib.sha256(z.read('mnist_dynamics/'+path)).hexdigest()
            assert actual==expected,path
        qa=json.loads(z.read('mnist_dynamics/results/qa.json'))
        assert qa['complete']
        manifest=json.loads(z.read('mnist_dynamics/configs/manifest.json'))
        run_ids={r['run_id'] for r in manifest}
        original=json.loads(z.read('mnist_dynamics/configs/manifest_original_42.json'))
        cancellation=json.loads(z.read('mnist_dynamics/metadata/user_cancellation.json'))
        original_ids={r['run_id'] for r in original}
        cancelled_ids={r['run_id'] for r in cancellation['cancelled_runs']}
        assert len(original)==len(original_ids)==42 and cancelled_ids<=original_ids
        assert hashlib.sha256(z.read('mnist_dynamics/configs/manifest_original_42.json')).hexdigest()==cancellation['original_manifest_sha256']
        assert run_ids==original_ids-cancelled_ids
        rows=list(csv.DictReader(io.StringIO(z.read('mnist_dynamics/results/runs.csv').decode('utf-8'))))
        assert len(manifest)==len(run_ids)==len(rows)==qa['expected_runs']==qa['completed_runs']
        assert {r['run_id'] for r in rows}==run_ids
        row_by_id={r['run_id']:r for r in rows}
        for entry in manifest:
            base='mnist_dynamics/'+entry['output']+'/'
            source=json.loads(z.read('mnist_dynamics/'+entry['config']))
            resolved=json.loads(z.read(base+'resolved_config.json'))
            status=json.loads(z.read(base+'status.json'))
            assert source==resolved and status['state']=='complete' and status['step']==100000
            digest=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert digest==status['config_hash']==row_by_id[entry['run_id']]['config_hash']
            for name in ('metrics.csv','norms.npy','events.json','train_indices.npy','environment.json','command.json'):
                assert entry['output']+'/'+name in hashes,entry['run_id']+' missing '+name
        assert len(z.read('log.md'))>0
        context=json.loads(z.read('mnist_dynamics/metadata/review_context_sha256.json'))
        for name,expected in context.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    print(json.dumps(dict(passed=True,hashed_files=len(hashes)+len(context),completed_runs=len(rows),cancelled_by_user=len(cancelled_ids))))


if __name__=='__main__':main()
