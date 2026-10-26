"""Combine three fresh run roots and preserve their configs."""
import argparse,csv,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]; configs=[]
    for seed,root in enumerate(args.runs):
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        status=json.loads((root/'status.json').read_text(encoding='utf-8'))
        assert config['seed']==seed and status['state']=='complete', root
        with (root/'results.csv').open(encoding='utf-8',newline='') as f: records=list(csv.DictReader(f))
        assert len(records)==630 and {int(r['seed']) for r in records}=={seed},root
        assert len({(int(r['n']),int(r['alpha_index'])) for r in records})==630,root
        rows.extend(records); configs.append(config)
    (args.output/'configs').mkdir(parents=True,exist_ok=True)
    for seed,config in enumerate(configs):
        (args.output/'configs'/('seed%d.json'%seed)).write_text(json.dumps(config,indent=2),encoding='utf-8')
    with (args.output/'all_seeds_results.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('Collected',len(rows),'rows into',args.output)

if __name__=='__main__': main()
