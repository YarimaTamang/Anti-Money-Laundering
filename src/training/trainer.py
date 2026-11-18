"""Reusable training / inference loop.

The trainer only ever receives explicit arrays of TRAIN and VALIDATION transaction ids;
test ids are never passed to ``fit`` (enforced in tests/test_training.py).
"""
from __future__ import annotations

import contextlib
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.training.losses import make_loss


class Prefetcher:
    """Builds batches in a background thread while the model computes."""

    def __init__(self, build: Callable[[np.ndarray], dict], batches: Iterable[np.ndarray], depth: int = 2):
        self.build, self.batches, self.depth = build, list(batches), depth

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=1) as ex:
            futs = [ex.submit(self.build, b) for b in self.batches[: self.depth]]
            nxt = self.depth
            while futs:
                f = futs.pop(0)
                if nxt < len(self.batches):
                    futs.append(ex.submit(self.build, self.batches[nxt]))
                    nxt += 1
                yield f.result()


def _chunks(a: np.ndarray, size: int):
    return [a[i:i + size] for i in range(0, len(a), size)]


def autocast_ctx(device: torch.device, amp: str | bool):
    if (amp is True or amp == "auto") and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def to_device(b: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


@torch.inference_mode()
def predict(model, store, eids: np.ndarray, cfg: dict, device=torch.device("cpu"), bs: int | None = None) -> np.ndarray:
    """Return raw logits for the given transaction ids (sigmoid applied only at evaluation)."""
    model.eval()
    bs = bs or cfg["eval_batch_size"]
    build = lambda e: store.build(e, model.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])  # noqa: E731
    out = []
    for b in Prefetcher(build, _chunks(np.asarray(eids), bs)):
        with autocast_ctx(device, cfg.get("amp", "auto")):
            out.append(model(to_device(b, device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def weighted_ap(y, logits, w) -> float:
    return float(average_precision_score(y, logits, sample_weight=w))


def fit(model, store, train_pos: np.ndarray, train_neg: np.ndarray, val_eids: np.ndarray, val_w: np.ndarray,
        cfg: dict, seed: int, log=print, device=torch.device("cpu")) -> tuple[torch.nn.Module, list[dict], dict]:
    rng = np.random.default_rng(seed)
    n_neg = min(cfg["neg_per_epoch"], len(train_neg))
    loss_fn = make_loss(cfg["loss"], len(train_pos), n_neg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=cfg["scheduler"]["factor"],
                                                     patience=cfg["scheduler"]["patience"])
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and cfg.get("amp", "auto") in ("auto", True) else None
    val_y = store.a["label"][val_eids].astype(np.float32)
    build = lambda e: store.build(e, model.needs, cfg["seq_len"], cfg["k1"], cfg["k2"])  # noqa: E731
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        eids = np.concatenate([train_pos, rng.choice(train_neg, n_neg, replace=False)])
        rng.shuffle(eids)
        tot, nb = 0.0, 0
        for b in Prefetcher(build, _chunks(eids, cfg["batch_size"])):
            b = to_device(b, device)
            with autocast_ctx(device, cfg.get("amp", "auto")):
                logits = model(b)
                loss = loss_fn(logits.float(), b["y"])
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                opt.step()
            tot += float(loss.detach())
            nb += 1
        t_train = time.time() - t0
        vl = predict(model, store, val_eids, cfg, device)
        with torch.no_grad():
            v_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(vl), torch.from_numpy(val_y), weight=torch.from_numpy(val_w.astype(np.float32))))
        v_ap = weighted_ap(val_y, vl, val_w)
        sch.step(v_ap)
        rec = {"epoch": epoch, "train_loss": tot / max(nb, 1), "val_loss": v_loss, "val_pr_auc": v_ap,
               "lr": opt.param_groups[0]["lr"], "epoch_train_s": t_train, "epoch_total_s": time.time() - t0}
        history.append(rec)
        log(f"epoch {epoch:02d} train_loss={rec['train_loss']:.4f} val_loss={v_loss:.4f} val_PR-AUC={v_ap:.4f} "
            f"lr={rec['lr']:.1e} ({rec['epoch_total_s']:.0f}s)")
        if v_ap > best:
            best, best_state, bad = v_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"early stopping at epoch {epoch} (best val PR-AUC {best:.4f})")
                break
    model.load_state_dict(best_state)
    return model, history, {"best_val_pr_auc": best, "best_epoch": int(np.argmax([h["val_pr_auc"] for h in history]) + 1)}
