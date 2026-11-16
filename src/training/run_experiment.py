"""Train + evaluate one experiment for one seed.

python -m src.training.run_experiment --exp tracex --seed 0 [--mode full|dev] [--max-epochs N]

Outputs:
  experiments/<mode>/<exp>/seed<k>/  config.yaml, history.csv, metrics.json,
                                     preds_test.npy (logits, full test split, eid order),
                                     preds_val_select.npz / preds_val_cal.npz (subsample logits + weights)
  checkpoints/<mode>/<exp>/seed<k>.pt
Selection (early stopping, decision threshold) uses val_select only; Platt scaling
uses val_cal only; the test split is predicted once, after selection.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocessing import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL_CAL, SPLIT_VAL_SELECT
from src.evaluation.alert_budget import alert_budget_metrics
from src.evaluation.calibration import PlattScaler, calibration_report
from src.evaluation.metrics import best_f1_threshold, classification_metrics, sigmoid
from src.graph.temporal_graph import TemporalGraphStore
from src.models.registry import build_model, count_parameters
from src.training.checkpoints import save_checkpoint
from src.training.trainer import fit, predict
from src.utils.config import PROJECT_ROOT, deep_update, load_yaml, processed_dir
from src.utils.logging import get_logger
from src.utils.reproducibility import get_device, set_seed, set_threads

warnings.filterwarnings("ignore")


def experiment_config(exp: str) -> tuple[str, dict, dict]:
    models = load_yaml("models")
    spec = models["experiments"][exp]
    params = deep_update(models.get("common_params", {}), spec.get("params", {}))
    train_cfg = deep_update(load_yaml("training"), spec.get("training", {}))
    return spec["arch"], params, train_cfg


def eval_subset(store: TemporalGraphStore, split_code: int, n_neg: int, seed: int):
    """All positives + fixed uniform negative sample; weights make it unbiased for the full split."""
    path = store.dir / f"eval_subset_{split_code}_{n_neg}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return z["eids"], z["w"]
    eids = store.split_eids(split_code)
    y = store.a["label"][eids]
    pos, neg = eids[y == 1], eids[y == 0]
    rng = np.random.default_rng(seed)
    neg_s = np.sort(rng.choice(neg, min(n_neg, len(neg)), replace=False))
    out = np.concatenate([pos, neg_s])
    order = np.argsort(out)
    out = out[order]
    w = np.where(store.a["label"][out] == 1, 1.0, len(neg) / len(neg_s))
    np.savez(path, eids=out, w=w)
    return out, w


def peak_memory_mb() -> float | None:
    try:
        import psutil

        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", getattr(mi, "rss", 0))) / 2**20
    except Exception:
        return None


def latency(model, store, cfg, eids, device) -> dict:
    """Single-transaction latency (batch building + forward), median over samples."""
    model.eval()
    ts = []
    with torch.inference_mode():
        for e in eids:
            t = time.perf_counter()
            b = store.build(np.array([e]), model.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])
            model({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()})
            ts.append(time.perf_counter() - t)
    return {"single_tx_latency_ms_median": float(np.median(ts) * 1000), "single_tx_latency_ms_p95": float(np.percentile(ts, 95) * 1000)}


def run(exp: str, seed: int, mode: str = "full", overrides: dict | None = None, out_root: str | Path | None = None) -> dict:
    arch, params, cfg = experiment_config(exp)
    cfg = deep_update(cfg, overrides or {})
    out_root = Path(out_root) if out_root else PROJECT_ROOT / "experiments" / mode
    out = out_root / exp / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    ck_path = PROJECT_ROOT / "checkpoints" / mode / exp / f"seed{seed}.pt"
    log = get_logger(f"{exp}.s{seed}", out / "train.log")
    set_seed(seed)
    set_threads(cfg.get("num_threads"))
    device = get_device(cfg.get("device", "auto"))
    store = TemporalGraphStore(processed_dir(load_yaml("data"), mode))
    dims = {"edge_dim": store.edge_dim, "node_dim": store.node_dim, "seq_dim": store.seq_dim}
    model = build_model(arch, dims, params).to(device)
    n_params = count_parameters(model)
    log.info(f"exp={exp} arch={arch} seed={seed} mode={mode} device={device} params={n_params} needs={model.needs}")
    with open(out / "config.yaml", "w") as f:
        yaml.safe_dump({"exp": exp, "arch": arch, "params": params, "training": cfg, "seed": seed, "mode": mode,
                        "dims": dims, "n_params": n_params}, f, sort_keys=False)

    split = store.a["split"]
    y_all = store.a["label"]
    tr = np.flatnonzero(split == SPLIT_TRAIN)
    train_pos, train_neg = tr[y_all[tr] == 1], tr[y_all[tr] == 0]
    vs_e, vs_w = eval_subset(store, SPLIT_VAL_SELECT, cfg["eval_subsample_neg"], cfg["eval_subsample_seed"])
    vc_e, vc_w = eval_subset(store, SPLIT_VAL_CAL, cfg["eval_subsample_neg"], cfg["eval_subsample_seed"])

    t0 = time.time()
    model, history, best = fit(model, store, train_pos, train_neg, vs_e, vs_w, cfg, seed, log.info, device)
    train_time = time.time() - t0
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    save_checkpoint(ck_path, model, arch, params, dims, {"exp": exp, "seed": seed, "mode": mode, "cfg": cfg})

    # ---- selection on val_select, calibration on val_cal (no test labels involved)
    vs_l = predict(model, store, vs_e, cfg, device)
    vc_l = predict(model, store, vc_e, cfg, device)
    vs_y, vc_y = y_all[vs_e].astype(float), y_all[vc_e].astype(float)
    thr, _ = best_f1_threshold(vs_y, sigmoid(vs_l), vs_w)
    platt = PlattScaler().fit(vc_l, vc_y, vc_w)
    np.savez(out / "preds_val_select.npz", eids=vs_e, logits=vs_l, w=vs_w)
    np.savez(out / "preds_val_cal.npz", eids=vc_e, logits=vc_l, w=vc_w)

    # ---- test: predicted once, after all choices are fixed
    te = store.split_eids(SPLIT_TEST)
    t1 = time.time()
    te_l = predict(model, store, te, cfg, device)
    test_infer_s = time.time() - t1
    np.save(out / "preds_test.npy", te_l.astype(np.float32))
    te_y = y_all[te].astype(float)
    te_p = sigmoid(te_l)
    te_pc = platt.transform(te_l)
    lat = latency(model, store, cfg, te[np.random.default_rng(0).choice(len(te), 50, replace=False)], device)
    metrics = {
        "exp": exp, "arch": arch, "seed": seed, "mode": mode, "n_params": n_params,
        "best_epoch": best["best_epoch"], "epochs_run": len(history),
        "val_select": classification_metrics(vs_y, sigmoid(vs_l), thr, vs_w),
        "test": classification_metrics(te_y, te_p, thr),
        "test_alert_budget": alert_budget_metrics(te_y, te_p),
        "test_calibration_raw": calibration_report(te_y, te_p),
        "test_calibration_platt": calibration_report(te_y, te_pc),
        "platt": {"a": platt.a, "b": platt.b},
        "efficiency": {"train_time_s": train_time, "test_inference_s": test_infer_s,
                       "test_throughput_tx_per_s": len(te) / max(test_infer_s, 1e-9),
                       "peak_memory_mb": peak_memory_mb(), **lat, "device": str(device),
                       "threads": torch.get_num_threads()},
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    t = metrics["test"]
    log.info(f"TEST PR-AUC={t['pr_auc']:.4f} ROC-AUC={t['roc_auc']:.4f} F1={t['f1']:.4f} P={t['precision']:.4f} "
             f"R={t['recall']:.4f} R@0.5%={metrics['test_alert_budget']['0.5%']['recall_at_k']:.4f} "
             f"train={train_time/60:.1f}min")
    if cfg.get("mlflow"):
        _mlflow_log(exp, seed, mode, params, cfg, metrics, ck_path, out)
    return metrics


def _mlflow_log(exp, seed, mode, params, cfg, metrics, ck_path, out):
    try:
        import mlflow

        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        mlflow.set_tracking_uri("sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix())
        mlflow.set_experiment(f"tracex-{mode}")
        with mlflow.start_run(run_name=f"{exp}-seed{seed}"):
            mlflow.log_params({"exp": exp, "seed": seed, **{f"p_{k}": v for k, v in params.items()},
                               **{f"t_{k}": v for k, v in cfg.items() if not isinstance(v, (dict, list))}})
            flat = {f"test_{k}": v for k, v in metrics["test"].items() if isinstance(v, float)}
            flat.update({f"val_{k}": v for k, v in metrics["val_select"].items() if isinstance(v, float)})
            flat["train_time_s"] = metrics["efficiency"]["train_time_s"]
            flat["recall_at_0.5pct"] = metrics["test_alert_budget"]["0.5%"]["recall_at_k"]
            mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in flat.items() if np.isfinite(v)})
            mlflow.set_tag("checkpoint", str(ck_path))
            mlflow.log_artifact(str(out / "metrics.json"))
    except Exception as e:  # tracking must never block training
        print(f"mlflow logging skipped: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="full", choices=["dev", "full"])
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--neg-per-epoch", type=int)
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    ov = {}
    if a.max_epochs:
        ov["max_epochs"] = a.max_epochs
    if a.neg_per_epoch:
        ov["neg_per_epoch"] = a.neg_per_epoch
    if a.threads:
        ov["num_threads"] = a.threads
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    run(a.exp, a.seed, a.mode, ov)
