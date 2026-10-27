"""Exercise scheduler atomic-write retries against a real Windows file lock."""
import ctypes
from ctypes import wintypes
import hashlib
import json
import threading
import time
from run_sweep import ROOT,atomic_json


def main():
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,
                                wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    kernel.CreateFileW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    probe=ROOT/'metadata/scheduler_io_probe.json'
    probe.write_text(json.dumps({'value':'before'}),encoding='utf-8')
    # Allow reads/writes, but deliberately deny deletion/replacement until release.
    handle=kernel.CreateFileW(str(probe),0x80000000,3,None,3,0,None)
    if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    timer=threading.Timer(.2,lambda:kernel.CloseHandle(handle))
    timer.start();start=time.perf_counter()
    try:retries=atomic_json(probe,{'value':'after','complete':True})
    finally:timer.join()
    elapsed=time.perf_counter()-start
    assert retries>0 and json.loads(probe.read_text())=={'value':'after','complete':True}
    probe.unlink()
    result=dict(passed=True,real_windows_sharing_lock=True,retries=retries,elapsed_seconds=elapsed,
                retry_limit=8,maximum_backoff_seconds=.5,
                training_sha256=hashlib.sha256((ROOT/'code/train_dynamics.py').read_bytes()).hexdigest())
    (ROOT/'metadata/scheduler_io_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
