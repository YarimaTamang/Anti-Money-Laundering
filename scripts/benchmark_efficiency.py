"""Controlled efficiency benchmark (run when the machine is otherwise idle).

Training wall-clock times in metrics.json were measured while two training jobs (and sometimes
analysis jobs) shared the CPU, so they are only indicative. This script measures every model
alone, in a fresh subprocess, with identical batches and thread count:
  * training throughput  (samples/s, forward + backward + AdamW step, batch 1024, incl. batch building)
  * inference throughput (transactions/s, batch 4096, incl. batch building)
  * single-transaction latency (median / p95 over 100 test transactions)
  * peak process memory (peak working set / RSS)
python scripts/benchmark_efficiency.py [--threads 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPS = ["lstm", "gru", "gcn", "graphsage", "gat", "gat_lstm", "tracex"]


def child(exp: str, threads: int) -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import psutil
    import torch

    from src.data.preprocessing import SPLIT_TEST, SPLIT_TRAIN
    from src.graph.temporal_graph import TemporalGraphStore
    from src.models.registry import build_model, count_parameters
    from src.training.run_experiment import experiment_config
    from src.utils.config import load_yaml, processed_dir

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    arch, params, cfg = experiment_config(exp)
    st = TemporalGraphStore(processed_dir(load_yaml("data"), "full"))
    dims = {"edge_dim": st.edge_dim, "node_dim": st.node_dim, "seq_dim": st.seq_dim}
    m = build_model(arch, dims, params)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    rng = np.random.default_rng(0)
    tr, te = st.split_eids(SPLIT_TRAIN), st.split_eids(SPLIT_TEST)
    build = lambda e: st.build(e, m.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])  # noqa: E731
    tb = [np.sort(rng.choice(tr, 1024, replace=False)) for _ in range(12)]
    m.train()
    for i, e in enumerate(tb):
        if i == 2:
            t0 = time.perf_counter()
        b = build(e)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(b), b["y"])
        opt.zero_grad(); loss.backward(); opt.step()
    train_sps = 1024 * 10 / (time.perf_counter() - t0)
    m.eval()
    ib = [te[i * 4096:(i + 1) * 4096] for i in range(8)]
    with torch.inference_mode():
        build(ib[0]); m(build(ib[0]))
        t0 = time.perf_counter()
        for e in ib:
            m(build(e))
        infer_tps = 4096 * len(ib) / (time.perf_counter() - t0)
        lat = []
        for e in rng.choice(te, 100, replace=False):
            t = time.perf_counter()
            m(build(np.array([e])))
            lat.append(time.perf_counter() - t)
    mi = psutil.Process().memory_info()
    return {"exp": exp, "params": count_parameters(m), "train_samples_per_s": train_sps, "infer_tx_per_s": infer_tps,
            "latency_ms_median": float(np.median(lat) * 1000), "latency_ms_p95": float(np.percentile(lat, 95) * 1000),
            "peak_memory_mb": float(getattr(mi, "peak_wset", mi.rss)) / 2**20, "threads": threads,
            "est_minutes_per_epoch_train": (cfg["neg_per_epoch"] + 2299) / train_sps / 60}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--child")
    a = ap.parse_args()
    if a.child:
        print("RESULT" + json.dumps(child(a.child, a.threads)))
        return
    rows = []
    for e in EXPS:
        r = subprocess.run([sys.executable, "-W", "ignore", __file__, "--child", e, "--threads", str(a.threads)],
                           capture_output=True, text=True, cwd=ROOT)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print(e, "FAILED", r.stderr[-2000:]); continue
        rows.append(json.loads(line[0][6:]))
        print(rows[-1], flush=True)
    out = ROOT / "reports" / "results"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out / "efficiency_benchmark.json", "w"), indent=1)
    from src.evaluation.comparison import DISPLAY

    md = [f"# Controlled efficiency benchmark (CPU, {a.threads} threads, one model at a time)", "",
          "| Model | Trainable params | Train samples/s | Est. min / epoch | Inference tx/s | Latency median (ms) | Latency p95 (ms) | Peak memory (MB) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {DISPLAY[r['exp']]} | {r['params']:,} | {r['train_samples_per_s']:,.0f} | {r['est_minutes_per_epoch_train']:.1f} | "
                  f"{r['infer_tx_per_s']:,.0f} | {r['latency_ms_median']:.1f} | {r['latency_ms_p95']:.1f} | {r['peak_memory_mb']:,.0f} |")
    md += ["", "Latency includes causal feature/subgraph construction for the transaction. Hardware: AMD Ryzen 7 5800H, 16 GB RAM, no CUDA GPU."]
    (out / "efficiency_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
