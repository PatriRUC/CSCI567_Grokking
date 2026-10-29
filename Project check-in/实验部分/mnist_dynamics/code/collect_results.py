"""Combine three fresh run roots: curves, configs and per-seed metadata."""
import argparse, csv, json, shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows, configs = [], []
    for seed, root in enumerate(args.runs):
        config = json.loads((root/'config.json').read_text(encoding='utf-8'))
        status = json.loads((root/'status.json').read_text(encoding='utf-8'))
        qa = json.loads((root/'qa_report.json').read_text(encoding='utf-8'))
        assert config['seed'] == seed and status['state'] == 'complete' and qa['complete'], root
        with (root/'curves.csv').open(encoding='utf-8', newline='') as f: records = list(csv.DictReader(f))
        assert len(records) == qa['models']*qa['evaluation_points'] and {int(r['seed']) for r in records} == {seed}, root
        rows.extend(records); configs.append(config)
    for name in ('configs', 'metadata'): (args.output/name).mkdir(parents=True, exist_ok=True)
    for seed, (root, config) in enumerate(zip(args.runs, configs)):
        (args.output/'configs'/('seed%d.json' % seed)).write_text(json.dumps(config, indent=2), encoding='utf-8')
        shutil.copyfile(root/'qa_report.json', args.output/'metadata'/('seed%d_qa.json' % seed))
        shutil.copyfile(root/'validation.json', args.output/'metadata'/('seed%d_validation.json' % seed))
        shutil.copyfile(root/'environment.json', args.output/'metadata'/('seed%d_environment.json' % seed))
    with (args.output/'all_seeds_curves.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Collected', len(rows), 'curve rows into', args.output)


if __name__ == '__main__': main()
