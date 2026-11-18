"""Evidence-based investigator summaries.

Every statement is generated from computed values (model output, verified transaction
records, account states computed strictly before the transaction, explanation outputs).
No statement asserts that laundering occurred.
"""
from __future__ import annotations

import numpy as np

DISCLAIMER = "Decision-support alert, not proof of money laundering."


GNN_DIFFUSE = ("Caution: no neighbouring transaction reached a GNNExplainer mask of 0.5 - the graph evidence for this "
               "score is diffuse, so the listed accounts should not be read as specific accomplices.")

def _cnt(v: float) -> int:
    return int(round(float(np.expm1(v))))


def account_profile(state: np.ndarray, names: list) -> dict:
    s = dict(zip(names, map(float, state)))
    has = s["has_history"] > 0.5
    return {
        "has_history": bool(has),
        "outgoing_count": _cnt(s["log_out_count"]), "incoming_count": _cnt(s["log_in_count"]),
        "outgoing_volume_usd": float(np.expm1(s["log_out_volume_usd"])),
        "incoming_volume_usd": float(np.expm1(s["log_in_volume_usd"])),
        "mean_outgoing_usd": float(np.expm1(s["log_mean_out_usd"])), "mean_incoming_usd": float(np.expm1(s["log_mean_in_usd"])),
        "unique_receivers": _cnt(s["log_unique_out_counterparties"]),
        "unique_senders": _cnt(s["log_unique_in_counterparties"]),
        "minutes_since_last_activity": _cnt(s["log_minutes_since_last"]) if has else None,
        "tx_per_hour": float(np.expm1(s["log_tx_per_hour"])),
    }


def build_summary(tx: dict, prob: float, prob_calibrated: float | None, sender: dict, receiver: dict,
                  gnn: dict | None, shap_res: dict | None, ig: dict | None, accounts: list | None = None,
                  uncertainty: dict | None = None, threshold: float | None = None) -> dict:
    name = (lambda a: accounts[a]) if accounts is not None else str
    findings = []
    if threshold is not None:
        findings.append(f"Model score {prob:.3f} is {'above' if prob >= threshold else 'below'} the validation-selected alert threshold {threshold:.3f}.")
    if shap_res:
        sv = np.asarray(shap_res["shap"])
        order = np.argsort(-np.abs(sv))[:5]
        top = [(shap_res["names"][i], float(sv[i])) for i in order]
        ups = [n for n, v in top if v > 0]
        downs = [n for n, v in top if v < 0]
        if ups:
            findings.append("Features that increased the score most (Kernel SHAP): " + ", ".join(ups) + ".")
        if downs:
            findings.append("Features that decreased the score (Kernel SHAP): " + ", ".join(downs) + ".")
    else:
        top = []
    if ig:
        steps = [("sender", r) for r in ig["sender_steps"]] + [("receiver", r) for r in ig["receiver_steps"]]
        steps.sort(key=lambda t: -t[1]["attribution"])
        pos = [t for t in steps if t[1]["attribution"] > 0][:3]
        if pos:
            desc = "; ".join(f"{side} history tx {r['tx_id']} ({r['minutes_before_target']} min earlier, attribution {r['attribution']:+.3f})" for side, r in pos)
            findings.append(f"Historical transactions contributing most towards the suspicious score (Integrated Gradients): {desc}.")
        findings.append(f"Total sequence attribution: sender history {ig['sender_total']:+.3f}, receiver history {ig['receiver_total']:+.3f} (logit units).")
    imp_acc = []
    if gnn:
        imp_acc = gnn["important_accounts"][:5]
        top_edges = sorted(gnn["edges"], key=lambda e: -e["importance"])[:5]
        if top_edges:
            findings.append("Most influential neighbouring transactions (GNNExplainer mask): " + ", ".join(
                f"tx {e['tx_id']} (hop {e['hop']}, mask {e['importance']:.2f})" for e in top_edges) + ".")
        ef = sorted(gnn["edge_feature_importance"].items(), key=lambda kv: -kv[1])[:3]
        findings.append("Edge attributes retained by the graph explanation: " + ", ".join(f"{k} ({v:.2f})" for k, v in ef) + ".")
        if gnn["edges"] and max(e["importance"] for e in gnn["edges"]) < 0.5:
            findings.append(GNN_DIFFUSE)
    if uncertainty:
        parts = []
        mc = uncertainty.get("mc_std")
        if mc is not None and np.isfinite(mc):
            parts.append(f"MC-dropout std {mc:.3f}")
        if uncertainty.get("conformal_outcome"):
            parts.append(f"conformal outcome: {uncertainty['conformal_outcome']}")
        if parts:
            findings.append("Uncertainty: " + "; ".join(parts) + ".")
    findings.append(DISCLAIMER)
    return {
        "graph_explanation_diffuse": bool(gnn and gnn["edges"] and max(e["importance"] for e in gnn["edges"]) < 0.5),
        "transaction": tx, "probability": prob, "probability_calibrated": prob_calibrated,
        "sender_profile": sender, "receiver_profile": receiver,
        "important_connected_accounts": [{"account": name(a["account"]), "importance": a["importance"]} for a in imp_acc],
        "top_features": top, "findings": findings, "disclaimer": DISCLAIMER,
    }


def to_markdown(s: dict) -> str:
    t = s["transaction"]
    sp, rp = s["sender_profile"], s["receiver_profile"]

    def prof(p):
        if not p["has_history"]:
            return "no earlier activity in the dataset"
        return (f"{p['outgoing_count']} outgoing / {p['incoming_count']} incoming earlier transactions, "
                f"{p['unique_receivers']} distinct receivers, {p['unique_senders']} distinct senders, "
                f"out-volume ${p['outgoing_volume_usd']:,.0f}, in-volume ${p['incoming_volume_usd']:,.0f}, "
                f"last activity {p['minutes_since_last_activity']} min before")

    cal = f" (Platt-calibrated: {s['probability_calibrated']:.4f})" if s.get("probability_calibrated") is not None else ""
    lines = [
        f"# TRACE-X investigation report — transaction {t['tx_id']}", "",
        f"> **{s['disclaimer']}**", "",
        "## Transaction", "",
        f"- Timestamp: {t['timestamp']}",
        f"- Sender: {t['sender']}  →  Receiver: {t['receiver']}",
        f"- Amount paid: {t['amount_paid']:,.2f} {t['currency_paid']} (≈ ${t['usd_paid']:,.2f}); received {t['amount_received']:,.2f} {t['currency_received']}",
        f"- Payment format: {t['payment_format']}",
        f"- Model suspicious score: **{s['probability']:.4f}**{cal}", "",
        "## Historical behaviour (strictly before this transaction)", "",
        f"- Sender: {prof(sp)}", f"- Receiver: {prof(rp)}", "",
    ]
    if s["important_connected_accounts"]:
        lines += ["## Important connected accounts (GNNExplainer)", ""]
        lines += [f"- {a['account']} — mask importance {a['importance']:.2f}" for a in s["important_connected_accounts"]]
        if s.get("graph_explanation_diffuse"):
            lines += ["", f"_{GNN_DIFFUSE}_"]
        lines.append("")
    if s["top_features"]:
        lines += ["## Feature contributions (Kernel SHAP, logit units)", "", "| Feature | SHAP |", "|---|---|"]
        lines += [f"| {n} | {v:+.3f} |" for n, v in s["top_features"]]
        lines.append("")
    lines += ["## Main explanation findings", ""] + [f"- {f}" for f in s["findings"]]
    if "ground_truth_label" in t:
        lines += ["", f"_Benchmark label (synthetic dataset, for evaluation only): {t['ground_truth_label']}_"]
    return "\n".join(lines)
