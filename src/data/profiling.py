"""Dataset profiling report: statistics (JSON + Markdown) and figures.

python -m src.data.profiling [--mode full]
Profiles the processed transactions table (identical content to the raw CSV plus derived
USD amounts) and also checks the raw CSV for duplicates / missing values.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.data.download import resolve_file
from src.graph.construction import graph_statistics
from src.utils.config import PROJECT_ROOT, load_yaml, processed_dir
from src.utils.plotting import NORMAL, SUSPICIOUS, apply_style, plt, save


def profile(mode: str = "full") -> dict:
    apply_style()
    cfg = load_yaml("data")
    pdir = processed_dir(cfg, mode)
    out = PROJECT_ROOT / "reports" / "profiling"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(pdir / "transactions.parquet")
    meta = json.load(open(pdir / "meta.json"))
    raw = pd.read_csv(resolve_file(cfg), dtype=str, nrows=None if mode == "full" else cfg["dev"]["n_rows"])
    y = df["label"].values
    n_acc = meta["n_accounts"]
    acc_tx = np.bincount(np.concatenate([df["src"].values, df["dst"].values]), minlength=n_acc)
    stats = {
        "mode": mode,
        "n_transactions": int(len(df)),
        "n_unique_sender_accounts": int(df["src"].nunique()),
        "n_unique_receiver_accounts": int(df["dst"].nunique()),
        "n_unique_accounts": int(n_acc),
        "n_unique_banks": int(pd.concat([df["from_bank"], df["to_bank"]]).nunique()),
        "missing_values_raw": int(raw.isna().sum().sum()),
        "duplicate_rows_raw": int(raw.duplicated().sum()),
        "n_laundering": int(y.sum()),
        "laundering_pct": float(100 * y.mean()),
        "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "distinct_timestamps": int(df["ts"].nunique()),
        "fraction_sharing_timestamp": float(1 - df["ts"].nunique() / len(df)),
        "usd_paid_quantiles": {q: float(df["usd_paid"].quantile(q)) for q in (0.01, 0.25, 0.5, 0.75, 0.99)},
        "usd_paid_median_by_label": {int(k): float(v) for k, v in df.groupby("label")["usd_paid"].median().items()},
        "payment_format": {k: {"count": int(v["count"]), "laundering_rate_pct": float(100 * v["mean"])}
                           for k, v in df.groupby("payment_format")["label"].agg(["count", "mean"]).iterrows()},
        "currency_paid": {k: int(v) for k, v in df["currency_paid"].value_counts().items()},
        "currency_mismatch_pct": float(100 * (df["cur_paid"] != df["cur_recv"]).mean()),
        "cross_bank_pct": float(100 * df["cross_bank"].mean()),
        "self_loop_pct": float(100 * df["self_loop"].mean()),
        "transactions_per_account_quantiles": {q: float(np.quantile(acc_tx, q)) for q in (0.5, 0.9, 0.99, 1.0)},
        "per_day": {str(d): {"n": int(g.size), "laundering": int(g.sum())}
                    for d, g in df.groupby(df["timestamp"].dt.date)["label"]},
        "splits": meta["split_counts"], "split_positives": meta["split_positives"],
        "split_time_ranges": meta["split_time_ranges"],
        "graph": graph_statistics(df["src"].values, df["dst"].values, n_acc),
    }
    with open(out / "profile.json", "w") as f:
        json.dump(stats, f, indent=2)

    # ---------------------------------------------------------------- figures
    # 1 class distribution
    fig, ax = plt.subplots(figsize=(5, 3))
    vals = [len(y) - y.sum(), y.sum()]
    ax.bar(["Normal", "Laundering"], vals, color=[NORMAL, SUSPICIOUS], width=0.5)
    ax.set_yscale("log"); ax.set_ylabel("transactions (log)")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.15, f"{v:,}\n({100*v/len(y):.3f}%)", ha="center", fontsize=8)
    ax.set_title("Class distribution"); ax.set_ylim(1, max(vals) * 8)
    save(fig, out / "class_distribution.png")

    # 2 amount distribution by label
    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.linspace(-2, 9, 80)
    for lab, c, name in ((0, NORMAL, "Normal"), (1, SUSPICIOUS, "Laundering")):
        v = np.log10(df.loc[df["label"] == lab, "usd_paid"].clip(lower=0.01))
        ax.hist(v, bins=bins, density=True, histtype="step", lw=2, color=c, label=name)
    ax.set_xlabel("log10(amount paid, USD)"); ax.set_ylabel("density"); ax.legend()
    ax.set_title("Transaction amount distribution (USD-converted)")
    save(fig, out / "amount_distribution.png")

    # 3 frequency distribution per account
    fig, ax = plt.subplots(figsize=(6, 3.2))
    cnt = np.bincount(acc_tx)
    k = np.flatnonzero(cnt)
    ax.loglog(k, cnt[k], "o", ms=3, color=NORMAL)
    ax.set_xlabel("transactions per account"); ax.set_ylabel("number of accounts")
    ax.set_title("Account transaction-frequency distribution")
    save(fig, out / "frequency_distribution.png")

    # 4 temporal distribution
    hourly = df.set_index("timestamp")["label"].resample("6h").agg(["size", "sum"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    axes[0].plot(hourly.index, hourly["size"], color=NORMAL); axes[0].set_ylabel("transactions / 6h")
    axes[0].set_title("Temporal distribution (6-hour bins)")
    axes[1].plot(hourly.index, hourly["sum"], color=SUSPICIOUS); axes[1].set_ylabel("laundering / 6h")
    # shaded split bands (alternating grey) labelled above the top panel, staggered so short splits don't collide
    for i, (name, (a, b)) in enumerate(meta["split_time_ranges"].items()):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        for ax in axes:
            ax.axvspan(a, b, color="#8a8984", alpha=0.10 if i % 2 == 0 else 0.22, lw=0)
        axes[0].annotate(name, xy=(a + (b - a) / 2, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(0, 3 + 9 * (i % 2)), textcoords="offset points", ha="center", va="bottom",
                         fontsize=7, color="#52514e")
    axes[0].set_title("Temporal distribution (6-hour bins) with chronological splits", pad=22)
    fig.autofmt_xdate()
    save(fig, out / "temporal_distribution.png")

    # 5 payment format
    pf = df.groupby("payment_format")["label"].agg(["size", "mean"]).sort_values("size")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    axes[0].barh(pf.index, pf["size"], color=NORMAL, height=0.6); axes[0].set_xlabel("transactions")
    axes[1].barh(pf.index, 100 * pf["mean"], color=SUSPICIOUS, height=0.6); axes[1].set_xlabel("laundering rate (%)")
    axes[0].set_title("Payment format: volume"); axes[1].set_title("Payment format: laundering rate")
    save(fig, out / "payment_format_distribution.png")

    # 6 currency
    cur = df.groupby("currency_paid")["label"].agg(["size", "mean"]).sort_values("size")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    axes[0].barh(cur.index, cur["size"], color=NORMAL, height=0.6); axes[0].set_xlabel("transactions")
    axes[1].barh(cur.index, 100 * cur["mean"], color=SUSPICIOUS, height=0.6); axes[1].set_xlabel("laundering rate (%)")
    axes[0].set_title("Payment currency: volume"); axes[1].set_title("Payment currency: laundering rate")
    save(fig, out / "currency_distribution.png")

    # 7 degree distribution
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for arr, c, name in ((np.bincount(df["src"].values, minlength=n_acc), NORMAL, "out-degree"),
                         (np.bincount(df["dst"].values, minlength=n_acc), SUSPICIOUS, "in-degree")):
        h = np.bincount(arr); k = np.flatnonzero(h); k = k[k > 0]
        ax.loglog(k, h[k], "o", ms=3, color=c, label=name)
    ax.set_xlabel("degree (transactions)"); ax.set_ylabel("accounts"); ax.legend()
    ax.set_title("Degree distribution of the transaction graph")
    save(fig, out / "degree_distribution.png")

    _markdown(stats, out)
    return stats


def _markdown(s: dict, out) -> None:
    g = s["graph"]
    lines = [
        f"# Dataset profile — IBM AML HI-Small ({s['mode']} mode)", "",
        "| Statistic | Value |", "|---|---|",
        f"| Transactions | {s['n_transactions']:,} |",
        f"| Unique sender accounts | {s['n_unique_sender_accounts']:,} |",
        f"| Unique receiver accounts | {s['n_unique_receiver_accounts']:,} |",
        f"| Unique accounts (bank+account) | {s['n_unique_accounts']:,} |",
        f"| Unique banks | {s['n_unique_banks']:,} |",
        f"| Missing values (raw) | {s['missing_values_raw']} |",
        f"| Exact duplicate rows (raw, kept) | {s['duplicate_rows_raw']} |",
        f"| Laundering transactions | {s['n_laundering']:,} ({s['laundering_pct']:.3f}%) |",
        f"| Time range | {s['time_range'][0]} → {s['time_range'][1]} |",
        f"| Distinct timestamps (minute resolution) | {s['distinct_timestamps']:,} |",
        f"| Cross-bank transactions | {s['cross_bank_pct']:.1f}% |",
        f"| Currency conversions | {s['currency_mismatch_pct']:.2f}% |",
        f"| Self-loop transactions | {s['self_loop_pct']:.1f}% |",
        f"| Median amount (USD) normal / laundering | {s['usd_paid_median_by_label'][0]:,.0f} / {s['usd_paid_median_by_label'][1]:,.0f} |",
        f"| Unique directed account pairs | {g['num_unique_directed_pairs']:,} |",
        f"| Reciprocal pair fraction | {g['reciprocal_pair_fraction']:.4f} |",
        f"| Weakly connected components | {g['weakly_connected_components']:,} (largest {g['largest_wcc_size']:,}) |",
        f"| Max out / in degree | {g['max_out_degree']:,} / {g['max_in_degree']:,} |", "",
        "## Payment formats", "", "| Format | Count | Laundering rate |", "|---|---|---|",
    ]
    for k, v in sorted(s["payment_format"].items(), key=lambda kv: -kv[1]["count"]):
        lines.append(f"| {k} | {v['count']:,} | {v['laundering_rate_pct']:.3f}% |")
    lines += ["", "## Transactions per day", "", "| Day | Transactions | Laundering |", "|---|---|---|"]
    for d, v in s["per_day"].items():
        lines.append(f"| {d} | {v['n']:,} | {v['laundering']:,} |")
    lines += ["", "## Chronological split", "", "| Split | Transactions | Laundering | Time range |", "|---|---|---|---|"]
    for k, v in s["splits"].items():
        tr = s["split_time_ranges"][k]
        lines.append(f"| {k} | {v:,} | {s['split_positives'][k]:,} | {tr[0][:16]} → {tr[1][:16]} |")
    lines += ["", "Figures: " + ", ".join(f"`{p}`" for p in [
        "class_distribution.png", "amount_distribution.png", "frequency_distribution.png", "temporal_distribution.png",
        "payment_format_distribution.png", "currency_distribution.png", "degree_distribution.png"])]
    (out / "profile.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    profile(ap.parse_args().mode)
