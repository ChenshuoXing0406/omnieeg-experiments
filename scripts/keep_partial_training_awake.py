"""Temporary system-awake request while these two experiment queues run; auto-release."""
import ctypes
from datetime import datetime
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
PATHS=[ROOT/'outputs/partial_sharing_v1/queue_status.json',ROOT/'outputs/partial_sharing_frozen_v1/queue_status.json']
STATUS=ROOT/'outputs/partial_sharing_v1/awake_guard_status.json'

def main():
    kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint]
    kernel.SetThreadExecutionState.restype=ctypes.c_uint
    assert kernel.SetThreadExecutionState(0x80000001),'System-awake request failed'
    start=time.monotonic()
    try:
        while time.monotonic()-start<4*3600:
            states=[json.loads(p.read_text())['status'] for p in PATHS]
            STATUS.write_text(json.dumps({'state':'active','queues':states,'local_time':datetime.now().isoformat(timespec='seconds'),
                'scope':'temporary system-awake request only; released after queues finish'}),encoding='utf-8')
            if all(s=='finished' for s in states) or any(s=='failed' for s in states):break
            time.sleep(20)
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        STATUS.write_text(json.dumps({'state':'released','local_time':datetime.now().isoformat(timespec='seconds')}),encoding='utf-8')

if __name__=='__main__':main()
