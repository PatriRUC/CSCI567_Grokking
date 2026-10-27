"""Create a portable local review bundle with light run evidence and hashes."""
import hashlib
import argparse
import csv
import json
import zipfile
from pathlib import Path
from project_scope import load_scope
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest,scope,_,_=load_scope()
    qa=json.loads((ROOT/'results/qa.json').read_text())
    assert qa['complete'] and qa['completed_runs']==scope['expected_runs']
    checkpoint_qa=json.loads((ROOT/'metadata/checkpoint_audit.json').read_text())
    assert checkpoint_qa['passed'] and checkpoint_qa['runs']==scope['expected_runs']
    process_qa=json.loads((ROOT/'metadata/process_outcomes.json').read_text(encoding='utf-8'))
    assert process_qa['runs']==scope['expected_runs'] and process_qa['artifacts_complete'] and process_qa['all_known_exit_codes_zero']
    pdf_qa=json.loads((ROOT/'metadata/pdf_export_validation.json').read_text())
    assert pdf_qa['render_passed'] and pdf_qa['visual_inspection_completed']
    for item in pdf_qa['pdfs']:assert sha(ROOT/'figures'/item['file'])==item['sha256']
    assert (ROOT/'results/REPORT.md').is_file()
    workspace=next(p for p in ROOT.parents if (p/'log.md').is_file())
    team=ROOT.parents[2]
    context={
        'log.md':workspace/'log.md',
        'repository_docs/README.md':team/'README.md',
        'repository_docs/Project check-in/README.md':team/'Project check-in/README.md',
        'repository_docs/Project check-in/实验部分/README.md':team/'Project check-in/实验部分/README.md',
        'execution_plan/PartC_Figure3de_Codex执行计划.md':ROOT.parents[3]/'PartC_Figure3de_Codex执行计划.md',
    }
    assert all(path.is_file() for path in context.values())
    (ROOT/'metadata/review_context_sha256.json').write_text(
        json.dumps({name:sha(path) for name,path in context.items()},indent=2,ensure_ascii=False),encoding='utf-8')
    with (ROOT/'results/runs.csv').open(encoding='utf-8',newline='') as f:run_rows=list(csv.DictReader(f))
    assert len(run_rows)==len(manifest) and {r['run_id'] for r in run_rows}=={r['run_id'] for r in manifest}
    cancelled=[]
    for run_id in scope['cancelled_run_ids']:
        folder=ROOT/'runs'/run_id
        artifacts=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),included_in_bundle=False)
                   for p in sorted(folder.rglob('*')) if p.is_file()] if folder.exists() else []
        cancelled.append(dict(run_id=run_id,counted_in_results=False,artifacts=artifacts))
    (ROOT/'metadata/cancelled_artifacts_index.json').write_text(json.dumps(cancelled,indent=2),encoding='utf-8')
    files=[ROOT/'README.md',ROOT/'.gitignore',ROOT/'.gitattributes']
    for directory in ['code','configs','results','figures','docs','logs']:
        files += [f for f in (ROOT/directory).rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']
    # Publication receipts contain this ZIP's digest and cannot be archived in it.
    files += [f for f in (ROOT/'metadata').rglob('*') if f.is_file() and f.suffix in ('.json','.txt','.log') and f.name!='publication.json']
    checkpoints=[]
    for entry in manifest:
        folder=ROOT/entry['output']
        files += [f for f in folder.iterdir() if f.is_file() and f.name!='norms.npy']
        # Retain all user-approved per-step norm traces (~3.2 MB each).
        files.append(folder/'norms.npy')
        for file in sorted((folder/'checkpoints').glob('*.pt')):
            checkpoints.append(dict(run_id=entry['run_id'],path=file.relative_to(ROOT).as_posix(),
                                    bytes=file.stat().st_size,sha256=sha(file),included_in_bundle=False))
    index=ROOT/'metadata/checkpoint_index.json'
    index.write_text(json.dumps(checkpoints,indent=2),encoding='utf-8');files.append(index)
    # Include the scientifically useful actual reference fit checkpoint.
    ref=ROOT/'runs/fig3d_n1000_a8_s0/checkpoints/fit.pt';files.append(ref)
    for item in checkpoints:
        if item['path']==ref.relative_to(ROOT).as_posix():item['included_in_bundle']=True
    index.write_text(json.dumps(checkpoints,indent=2),encoding='utf-8')
    files.append(ROOT/'PUBLICATION.md')
    unique=sorted(set(files))
    hashes={f.relative_to(ROOT).as_posix():sha(f) for f in unique}
    (ROOT/'manifest_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    unique.append(ROOT/'manifest_sha256.json')
    destination=args.output.resolve() if args.output else ROOT.parents[3]/'PartC_Figure3de_本地审阅包.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in unique:z.write(file,arcname='mnist_dynamics/'+file.relative_to(ROOT).as_posix())
        for name,file in context.items():z.write(file,arcname=name)
    with zipfile.ZipFile(destination) as z:assert z.testzip() is None
    print(json.dumps(dict(bundle=str(destination),bytes=destination.stat().st_size,files=len(unique)+len(context),sha256=sha(destination)),ensure_ascii=False))


if __name__=='__main__':main()
