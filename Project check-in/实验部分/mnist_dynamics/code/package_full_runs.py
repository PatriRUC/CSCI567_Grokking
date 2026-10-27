"""Archive all retained raw runs, or verify that archive, using the standard library."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from project_scope import load_scope

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'mnist_dynamics/'


def sha_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def sha_file(path):
    with path.open('rb') as stream:
        return sha_stream(stream)


def verify(path):
    with zipfile.ZipFile(path) as archive:
        index = json.loads(archive.read('full_runs_manifest.json'))
        files = index['files']
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == {PREFIX + name for name in files} | {'full_runs_manifest.json'}
        for name, info in files.items():
            member = archive.getinfo(PREFIX + name)
            assert member.file_size == info['bytes'], name
            with archive.open(member) as stream:
                assert sha_stream(stream) == info['sha256'], name
        manifest = json.loads(archive.read(PREFIX + 'configs/manifest.json'))
        original = json.loads(archive.read(PREFIX + 'configs/manifest_original_42.json'))
        cancellation = json.loads(archive.read(PREFIX + 'metadata/user_cancellation.json'))
        original_bytes = archive.read(PREFIX + 'configs/manifest_original_42.json')
        assert hashlib.sha256(original_bytes).hexdigest() == cancellation['original_manifest_sha256']
        cancelled = {entry['run_id'] for entry in cancellation['cancelled_runs']}
        ids = {entry['run_id'] for entry in manifest}
        assert len(original) == 42 and len(ids) == len(manifest) == 34
        assert ids == {entry['run_id'] for entry in original} - cancelled
        assert set(index['retained_run_ids']) == ids
        for entry in manifest:
            base = entry['output'] + '/'
            source = json.loads(archive.read(PREFIX + entry['config']))
            resolved = json.loads(archive.read(PREFIX + base + 'resolved_config.json'))
            status = json.loads(archive.read(PREFIX + base + 'status.json'))
            assert source == resolved and status['state'] == 'complete' and status['step'] == 100000
            digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            assert status['config_hash'] == digest
            for required in ('metrics.csv', 'norms.npy', 'train_indices.npy', 'events.json', 'command.json', 'environment.json'):
                assert base + required in files
            for checkpoint in ('init.pt', 'fit.pt', 'latest.pt', 'final.pt'):
                assert base + 'checkpoints/' + checkpoint in files
        count = sum(name.endswith('.pt') for name in files)
        assert count == index['checkpoints'] == 136
    return dict(passed=True, retained_runs=34, cancelled_by_user=len(cancelled),
                checkpoints=count, hashed_files=len(files))


def create(destination):
    manifest, scope, _, _ = load_scope()
    assert scope['expected_runs'] == 34
    destination = destination.resolve()
    assert ROOT not in destination.parents, 'Write large archives outside the repository Part C folder.'
    assert not destination.exists(), 'Choose a new output path; existing archives are preserved.'
    files = []
    for entry in manifest:
        folder = ROOT / entry['output']
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        assert status['state'] == 'complete' and status['step'] == 100000
        files.extend(path for path in folder.rglob('*') if path.is_file())
    files.extend(path for path in (ROOT / 'configs').glob('*.json'))
    for name in ('user_cancellation.json', 'checkpoint_audit.json', 'provenance.json'):
        files.append(ROOT / 'metadata' / name)
    files.extend([ROOT / 'code/train_dynamics.py', ROOT / 'code/project_scope.py'])
    unique = sorted(set(files))
    index = dict(format_version=1, retained_run_ids=[entry['run_id'] for entry in manifest],
                 cancelled_run_ids=scope['cancelled_run_ids'], checkpoints=136,
                 files={path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha_file(path))
                        for path in unique})
    (ROOT / 'metadata/full_runs_index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('full_runs_manifest.json', json.dumps(index, ensure_ascii=False, indent=2))
        for path in unique:
            archive.write(path, PREFIX + path.relative_to(ROOT).as_posix())
    result = verify(destination)
    result.update(bundle=str(destination), bytes=destination.stat().st_size, sha256=sha_file(destination))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else create(args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
