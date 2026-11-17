"""Run (experiment, seed) pairs sequentially, skipping completed runs (metrics.json present).

python scripts/run_queue.py --exps tracex,lstm,gat --seeds 0,1,2 [--mode full] [--threads 8]
Each run executes in a fresh subprocess so a crash cannot take down the queue.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--mode", default="full")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed-major", action="store_true", help="iterate seeds in the outer loop")
    a = ap.parse_args()
    exps, seeds = a.exps.split(","), [int(s) for s in a.seeds.split(",")]
    jobs = [(e, s) for s in seeds for e in exps] if a.seed_major else [(e, s) for e in exps for s in seeds]
    for exp, seed in jobs:
        run_dir = ROOT / "experiments" / a.mode / exp / f"seed{seed}"
        if (run_dir / "metrics.json").exists():
            print(f"[skip] {exp} seed{seed}", flush=True)
            continue
        # atomic claim so several queue workers can share one job list
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(run_dir / ".claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
        except FileExistsError:
            print(f"[skip] {exp} seed{seed} (claimed by another worker)", flush=True)
            continue
        cmd = [sys.executable, "-W", "ignore", "-m", "src.training.run_experiment", "--exp", exp, "--seed", str(seed), "--mode", a.mode]
        if a.threads:
            cmd += ["--threads", str(a.threads)]
        t = time.time()
        print(f"[run ] {exp} seed{seed}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {exp} seed{seed} rc={r.returncode} {(time.time()-t)/60:.1f}min", flush=True)
        if r.returncode != 0:
            (run_dir / ".claim").unlink(missing_ok=True)  # allow a retry


if __name__ == "__main__":
    main()
