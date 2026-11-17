"""Keep Windows awake (system + display) while experiment queues are running, then release.

Uses SetThreadExecutionState (no persistent settings change). Exits - and releases the
request - when no `run_queue.py` / `run_experiment` / analysis process is alive for 5 minutes.
python scripts/keep_awake.py
"""
import ctypes
import sys
import time

import psutil

ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED, ES_AWAYMODE_REQUIRED = 0x80000000, 0x1, 0x2, 0x40
MARKERS = ("run_queue.py", "run_experiment", "explain.py", "uncertainty.py", "benchmark_efficiency.py")


def busy() -> bool:
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if any(m in cmd for m in MARKERS) and "keep_awake" not in cmd:
            return True
    return False


def main():
    if sys.platform != "win32":
        print("keep_awake is Windows-only"); return
    k = ctypes.windll.kernel32
    idle_since = None
    while True:
        k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED)
        if busy():
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 300:
                break
        time.sleep(30)
    k.SetThreadExecutionState(ES_CONTINUOUS)
    print("released keep-awake request", flush=True)


if __name__ == "__main__":
    main()
