"""Temporal leakage-prevention tests (brute-force references + real cached data)."""
import json

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import SPLIT_TEST, SPLIT_TRAIN, chronological_split
from src.graph.construction import build_static_graph, snapshot_before, temporal_neighbors
from src.graph.features import EDGE_CONTINUOUS, NODE_FEATURES, node_states
from src.sequences.receiver_history import receiver_history
from src.sequences.sender_history import sender_history


# ----------------------------------------------------------------- synthetic
def test_sender_history_matches_bruteforce(synth):
    s, ts, L = synth["src"], synth["ts"], 6
    h = sender_history(s, ts, L)
    for e in range(synth["n"]):
        prev = [j for j in range(synth["n"]) if s[j] == s[e] and ts[j] < ts[e]]
        prev = sorted(prev, key=lambda j: (ts[j], j))[-L:]
        got = [int(v) for v in h[e] if v >= 0]
        assert got == prev, e
        assert all(ts[j] < ts[e] for j in got)


def test_receiver_history_matches_bruteforce(synth):
    d, ts, L = synth["dst"], synth["ts"], 6
    h = receiver_history(d, ts, L)
    for e in range(synth["n"]):
        prev = sorted([j for j in range(synth["n"]) if d[j] == d[e] and ts[j] < ts[e]], key=lambda j: (ts[j], j))[-L:]
        assert [int(v) for v in h[e] if v >= 0] == prev


def test_same_timestamp_transactions_never_leak(synth):
    s, d, ts = synth["src"], synth["dst"], synth["ts"]
    h = sender_history(s, ts, 20)
    nb = temporal_neighbors(s, d, ts, 20)
    for e in range(synth["n"]):
        for j in h[e][h[e] >= 0]:
            assert ts[j] != ts[e]
        codes = nb[e][nb[e] >= 0]
        assert np.all(ts[codes >> 1] < ts[e])


def test_temporal_neighbors_match_bruteforce(synth):
    s, d, ts, n, K = synth["src"], synth["dst"], synth["ts"], synth["n"], 5
    nb = temporal_neighbors(s, d, ts, K)
    for e in range(n):
        for side, acct in ((0, s[e]), (1, d[e])):
            ev = []
            for j in range(n):
                if ts[j] >= ts[e]:
                    continue
                if s[j] == acct:
                    ev.append((ts[j], j, 0, 2 * j + 1))  # counterparty is receiver
                if d[j] == acct:
                    ev.append((ts[j], j, 1, 2 * j + 0))
            want = [c for *_, c in sorted(ev)[-K:]]
            got = [int(c) for c in nb[e, side] if c >= 0]
            assert got == want, (e, side)


def test_node_states_match_bruteforce(synth):
    s, d, ts, usd, n = synth["src"], synth["dst"], synth["ts"], synth["usd"], synth["n"]
    st = node_states(s, d, ts, usd, usd)
    fi = {f: i for i, f in enumerate(NODE_FEATURES)}
    for e in range(0, n, 7):
        for side, acct in ((0, s[e]), (1, d[e])):
            past = [j for j in range(n) if ts[j] < ts[e]]
            out = [j for j in past if s[j] == acct]
            inn = [j for j in past if d[j] == acct]
            assert np.isclose(st[e, side, fi["log_out_count"]], np.log1p(len(out)))
            assert np.isclose(st[e, side, fi["log_in_count"]], np.log1p(len(inn)))
            assert np.isclose(st[e, side, fi["log_out_volume_usd"]], np.log1p(usd[out].sum()), rtol=1e-5)
            assert np.isclose(st[e, side, fi["log_unique_out_counterparties"]], np.log1p(len(set(d[out]))))
            assert np.isclose(st[e, side, fi["log_unique_in_counterparties"]], np.log1p(len(set(s[inn]))))
            act = out + inn
            assert st[e, side, fi["has_history"]] == float(len(act) > 0)
            if act:
                assert np.isclose(st[e, side, fi["log_minutes_since_last"]], np.log1p(ts[e] - max(ts[act])))


def test_snapshot_excludes_future_edges(synth):
    g = build_static_graph(synth["src"], synth["dst"], synth["ts"], np.arange(synth["n"]))
    for t in (0, 10, 25, 39):
        snap = snapshot_before(g, t)
        assert int((snap.edge_ts >= t).sum()) == 0
        assert snap.edge_index.shape[1] == int((torch.as_tensor(synth["ts"]) < t).sum())


def test_chronological_split_does_not_split_timestamps(synth):
    ts = synth["ts"]
    sp = chronological_split(ts, 0.6, 0.2, 0.5)
    assert np.all(np.diff(sp) >= 0), "split codes must be monotone in time"
    for a in np.unique(sp):
        for b in np.unique(sp):
            if a < b:
                assert ts[sp == a].max() < ts[sp == b].min()


# --------------------------------------------------------------- real (dev)
def test_store_batches_are_causal(store):
    rng = np.random.default_rng(1)
    eids = rng.choice(store.n, 3000, replace=False)
    b = store.build(eids, needs=("seq", "graph"), seq_len=20, k1=10, k2=5, meta=True)
    ts = store.a["ts"]
    tgt_t = ts[eids]
    for key in ("seq_s_eid", "seq_r_eid"):
        idx = b[key].numpy()
        m = idx >= 0
        assert np.all(ts[idx[m]] < np.broadcast_to(tgt_t[:, None], idx.shape)[m])
    # graph: hop-1 edges strictly before the target, hop-2 strictly before their hop-1 parent edge
    ei = b["g_edge_index"].numpy()
    e_eid, hop = b["g_edge_eid"].numpy(), b["g_edge_hop"].numpy()
    nb = len(eids)
    parent = ei[1]
    hop1 = hop == 1
    assert np.all(ts[e_eid[hop1]] < tgt_t[parent[hop1] // 2])
    edge_of_node = {}
    for j in np.flatnonzero(hop1):
        edge_of_node[ei[0, j]] = e_eid[j]
    for j in np.flatnonzero(~hop1):
        assert ts[e_eid[j]] < ts[edge_of_node[parent[j]]]
    assert np.all(b["g_edge_dt"].numpy() > 0)
    assert b["g_center_index"].shape == (nb, 2)


def test_sequence_and_neighbor_tables_are_causal_full_scan(store):
    ts = store.a["ts"]
    for key in ("hist_s", "hist_r"):
        h = np.asarray(store.a[key])
        m = h >= 0
        assert np.all(ts[h[m]] < np.broadcast_to(ts[:, None], h.shape)[m]), key
    nb = np.asarray(store.a["nbr"])
    m = nb >= 0
    assert np.all(ts[nb[m] >> 1] < np.broadcast_to(ts[:, None, None], nb.shape)[m])


def test_label_excluded_from_features(store):
    names = store.edge_names + store.node_names
    assert not any("label" in n.lower() or "launder" in n.lower() for n in names)
    y = store.a["label"].astype(np.float32)
    ef = np.asarray(store.a["edge_feat"])
    for j in range(ef.shape[1]):
        assert not np.array_equal(ef[:, j], y)
    b = store.build(np.arange(100), needs=("seq", "graph"))
    feature_keys = [k for k in b if k not in ("y", "eid")]
    assert "label" not in feature_keys


def test_scalers_fit_on_training_rows_only(store):
    split = store.a["split"]
    tr = split == SPLIT_TRAIN
    ef = np.asarray(store.a["edge_feat"])
    for j in EDGE_CONTINUOUS:
        assert np.isclose(store.edge_mean[j], ef[tr, j].mean(), rtol=1e-4)
    ns = np.asarray(store.a["node_state"])[tr].reshape(-1, len(store.node_names))
    assert np.allclose(store.node_mean[:-1], ns[:, :-1].mean(0), rtol=1e-3, atol=1e-4)
    assert store.stats["fitted_on"] == "train"
    assert store.stats["n_train_rows"] == int(tr.sum())


def test_splits_are_chronological_and_disjoint(store):
    ts, split = store.a["ts"], store.a["split"]
    assert np.all(np.diff(split) >= 0)
    for a in range(SPLIT_TEST):
        if (split == a).any() and (split == a + 1).any():
            assert ts[split == a].max() < ts[split == a + 1].min()
    frac_train = (split == SPLIT_TRAIN).mean()
    assert 0.58 < frac_train < 0.63
