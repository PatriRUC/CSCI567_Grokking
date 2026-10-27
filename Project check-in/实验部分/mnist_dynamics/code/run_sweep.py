"""Execute a frozen manifest locally with per-run logs and checkpoint resume."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from train_dynamics import ROOT, atomic_json as training_atomic_json, config_hash


def atomic_json(path,value):
    """Retry bounded Windows sharing conflicts in scheduler metadata writes."""
    for attempt in range(8):
        try:
            training_atomic_json(path,value)
            return attempt
        except PermissionError as exc:
            if getattr(exc,'winerror',None) not in (5,32,33) or attempt==7:raise
            time.sleep(min(.05*2**attempt,.5))


def append_work_log(message):
    workspace = next((p for p in ROOT.parents if (p/'log.md').is_file()), None)
    if workspace:
        stamp=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')
        with (workspace/'log.md').open('a',encoding='utf-8') as f:
            f.write(f'\n- {stamp}：{message}\n')


def user_cancelled_ids():
    record=ROOT/'metadata/user_cancellation.json'
    if not record.exists():return set()
    return {entry['run_id'] for entry in json.loads(record.read_text(encoding='utf-8-sig')).get('cancelled_runs',[])}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/manifest.json')
    p.add_argument('--only',nargs='*')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--max-concurrent',type=int,default=1)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    cancelled=user_cancelled_ids()
    omitted=[entry['run_id'] for entry in manifest if entry['run_id'] in cancelled]
    if omitted:print('SKIP_USER_CANCELLED',','.join(omitted),flush=True)
    manifest=[entry for entry in manifest if entry['run_id'] not in cancelled]
    if args.only:manifest=[r for r in manifest if r['run_id'] in args.only]
    if args.dry_run:
        print(json.dumps(manifest,indent=2));return
    validation=json.loads((ROOT/'metadata/validation/validation.json').read_text())
    if not validation['passed']:raise RuntimeError('Validation not passed')
    if args.max_concurrent not in (1,4):
        raise ValueError('Supported concurrency is 1 or validated 4.')
    if args.max_concurrent==4:
        check=json.loads((ROOT/'metadata/concurrency4/validation.json').read_text())
        assert check['passed'] and check['workers']==4
        # A previously safe four-worker run may no longer fit a shared laptop GPU.
        try:
            free_mb=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                                                text=True,timeout=10).strip().splitlines()[0])
        except (OSError,ValueError,subprocess.SubprocessError):
            free_mb=None
        if free_mb is None or free_mb<5500:
            args.max_concurrent=1
            append_work_log(f'请求四并发时GPU空闲显存为{free_mb} MiB，低于保守启动余量5500 MiB；自动改为串行，保留相同配置。')
            print('MEMORY_GUARD_USING_SERIAL',free_mb,flush=True)
    pending=[]
    for entry in manifest:
        out=ROOT/entry['output']
        config=json.loads((ROOT/entry['config']).read_text())
        if (out/'status.json').exists():
            status=json.loads((out/'status.json').read_text())
            if status['state']=='complete':
                assert status['config_hash']==config_hash(config)
                print('SKIP_COMPLETE',entry['run_id'],flush=True);continue
        pending.append(entry)
    active=[];completed=[];failures=[]
    def launch(entry):
        if entry['run_id'] in user_cancelled_ids():
            print('SKIP_USER_CANCELLED',entry['run_id'],flush=True)
            append_work_log(f"按用户取消记录跳过 `{entry['run_id']}`，未启动训练。")
            return
        out=ROOT/entry['output']
        config=json.loads((ROOT/entry['config']).read_text())
        out.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(ROOT/'code/train_dynamics.py')]
        if (out/'checkpoints/latest.pt').exists():
            previous=json.loads((out/'resolved_config.json').read_text())
            if previous!=config:raise RuntimeError('Refuse resume under a different manifest configuration')
            command += ['--resume',str(out/'checkpoints/latest.pt')]
        else:command += ['--config',str(ROOT/entry['config']),'--output',str(out)]
        atomic_json(out/'command.json',dict(argv=command))
        print('START',entry['run_id'],flush=True)
        append_work_log(f"开始本地运行 `{entry['run_id']}`；命令见 `{entry['output']}/command.json`，stdout/stderr 见 console.log。")
        handle=(out/'console.log').open('a',encoding='utf-8')
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
        active.append((entry,process,handle))
    while pending or active:
        while pending and len(active)<args.max_concurrent and not failures:
            launch(pending.pop(0))
        for entry,process,handle in list(active):
            rc=process.poll()
            if rc is None:continue
            out=ROOT/entry['output']
            atomic_json(out/'process_exit.json',dict(run_id=entry['run_id'],pid=process.pid,exit_code=rc,
                        finished_at=datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(timespec='seconds')))
            handle.close();active.remove((entry,process,handle))
            if rc:
                failures.append(dict(run_id=entry['run_id'],exit_code=rc))
                append_work_log(f"运行 `{entry['run_id']}` 失败，退出码 {rc}；保留日志与检查点等待排查。")
                continue
            status=json.loads((out/'status.json').read_text())
            events=json.loads((out/'events.json').read_text())
            assert status['state']=='complete'
            completed.append(entry['run_id'])
            print('COMPLETE',entry['run_id'],flush=True)
            append_work_log(f"完成 `{entry['run_id']}`，{status['step']} steps，进程内计时 {status['elapsed_seconds']:.1f} 秒；t60={events['t60']}，拟合 step={events['fit_step']}。这是单次结果，最终统一汇总。")
        atomic_json(ROOT/'metadata/sweep_status.json',dict(state='running' if not failures else 'failure_draining',
                    active=[e['run_id'] for e,p,h in active],completed_this_invocation=completed,
                    pending=[e['run_id'] for e in pending],failures=failures,concurrency=args.max_concurrent))
        if failures and not active:
            raise RuntimeError(f'Runs failed: {failures}; remaining runs were not started.')
        if active:time.sleep(2)
    atomic_json(ROOT/'metadata/sweep_status.json',dict(state='complete',runs=len(manifest)))


if __name__=='__main__':main()
