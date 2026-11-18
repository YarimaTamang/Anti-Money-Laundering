# TRACE-X: technical completion report

Date: 18 Nov 2026 · Commit: see `git log` · Dataset: IBM AML HI-Small, **full mode**, 5,078,345 transactions.

This report lists what was actually executed and verified. Anything that was not executed is marked as such.

## A. Implementation status

| Component | Implemented | Tested / verified |
|---|---|---|
| Dataset download (kagglehub / local file / env var) | ✅ | ✅ downloaded and loaded; loading test |
| Profiling report and 7 figures | ✅ | ✅ `reports/profiling/` |
| Preprocessing (account IDs, train-only FX rates, encodings), dev and full modes | ✅ | ✅ both modes run; tests |
| Chronological 60/10/10/20 split snapped to timestamp changes | ✅ | ✅ test |
| Causal index; edge (27) and account-state (11) features computed strictly before t | ✅ | ✅ 12 leakage tests against brute force |
| Sender and receiver history sequences | ✅ | ✅ tests |
| Causal 2-hop temporal neighbourhood (TemporalGraphStore) | ✅ | ✅ tests, including the depth-pruning equivalence |
| TRACE-X (EdgeTemporalGATConv, sender/receiver LSTMs, time encoding, temporal branch, gated fusion) | ✅ | ✅ shape and forward/backward tests; trained |
| Baselines: LSTM, GRU, GCN, GraphSAGE, GAT, GAT-LSTM | ✅ | ✅ tests; trained |
| 10 ablation variants | ✅ | ✅ forward/backward tests; trained |
| Training: AdamW, weighted BCE (focal optional), ReduceLROnPlateau, early stopping, grad clipping, seeds, checkpoints, YAML configs | ✅ | ✅ |
| AMP | ✅ CUDA only | ⚠️ not exercised: no CUDA GPU (CPU autocast deliberately disabled) |
| Focal loss | ✅ | ⚠️ unit-level only; no full experiment (`tracex_focal` config exists) |
| Evaluation: P/R/F1/ROC-AUC/PR-AUC/confusion matrix, alert budget, calibration, efficiency | ✅ | ✅ |
| GNNExplainer (adapted), Kernel SHAP, Integrated Gradients | ✅ | ✅ tests (validity, causality, additivity, completeness) |
| Fidelity / sparsity / stability metrics | ✅ | ✅ run on 10 cases, stability on 3 seeds |
| Investigator reports (Markdown/HTML) | ✅ | ✅ 10 cases plus dashboard generation |
| MC Dropout, class-conditional split conformal, 3-way triage | ✅ | ✅ run on TRACE-X seed 0 |
| Streamlit dashboard (6 pages) | ✅ | ✅ all pages rendered and screenshotted |
| MLflow tracking (SQLite) | ✅ | ✅ every run logged |
| FastAPI `/health`, `/predict`, `/explain` | ✅ | ✅ TestClient smoke test |
| Dockerfile / Docker image | ✅ | ✅ built (Docker 29.9.3); dashboard container (6 pages) and API container verified |
| Optional TGN extension | ❌ | — not implemented (optional) |
| Demo video | ❌ | — not produced |

## B. Model training status

All runs use full mode. Checkpoints are in `checkpoints/full/<exp>/seed<k>.pt` (git-ignored). Per-run outputs are in `experiments/full/<exp>/seed<k>/`.

| Model | Implemented | Trained | Seeds completed |
|---|---|---|---|
| TRACE-X | ✅ | ✅ | 3 (0, 1, 2) |
| LSTM | ✅ | ✅ | 3 |
| GRU | ✅ | ✅ | 3 |
| GCN | ✅ | ✅ | 3 |
| GraphSAGE | ✅ | ✅ | 3 |
| GAT | ✅ | ✅ | 3 |
| GAT-LSTM | ✅ | ✅ | 3 |
| Ablations: sequence-only, graph-only, no temporal encoding, concat fusion | ✅ | ✅ | 2 (0, 1) |
| Ablations: no edge features, mean aggregation, no sender, no receiver, seq length 5, seq length 20 | ✅ | ✅ | 1 (0) |

Notes on interrupted runs:

- **abl_graph_only seed 1** was killed after training had finished. It was evaluated from its saved best checkpoint with `--eval-only`. The training time comes from its per-epoch history.
- **abl_no_temporal_enc seed 1** was killed mid-training and fully retrained.
- **TRACE-X seed 0 and LSTM seed 1** were paused by system sleep. Their wall-clock training times are excluded (`TIMING_INVALID.txt`); their metrics are unaffected.

## C. Actual performance results

Full test split (1,015,564 transactions, 1,797 laundering). Mean ± std over 3 seeds.

- Precision, recall and F1 use the F1-optimal threshold chosen on val_select.
- K = 0.5 % of test transactions.
- Training time is wall-clock, often with two jobs sharing the CPU, so it is indicative only.
- Latency is from the controlled benchmark: one transaction including causal batch construction, 8 threads, median.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | P@K | R@K | Train min | Latency ms |
|---|---|---|---|---|---|---|---|---|---|
| LSTM | 0.9873 ± 0.0001 | 0.3907 ± 0.0085 | 0.3311 ± 0.0306 | 0.4485 ± 0.0256 | 0.3790 ± 0.0137 | 0.2041 ± 0.0019 | 0.5769 ± 0.0054 | 3.7 | 6.1 |
| GRU | 0.9878 ± 0.0004 | 0.3810 ± 0.0142 | 0.4043 ± 0.0152 | 0.3882 ± 0.0204 | 0.3957 ± 0.0134 | 0.2099 ± 0.0026 | 0.5932 ± 0.0075 | 3.7 | 3.4 |
| GCN | 0.9845 ± 0.0006 | 0.2764 ± 0.0123 | 0.2768 ± 0.0167 | 0.4389 ± 0.0561 | 0.3368 ± 0.0090 | 0.1998 ± 0.0075 | 0.5646 ± 0.0211 | 57.7 | 8.7 |
| GraphSAGE | 0.9860 ± 0.0002 | 0.2751 ± 0.0159 | 0.2796 ± 0.0493 | 0.4786 ± 0.0829 | 0.3428 ± 0.0101 | 0.2116 ± 0.0011 | 0.5980 ± 0.0030 | 58.6 | 2.9 |
| GAT | 0.9843 ± 0.0008 | 0.2636 ± 0.0345 | 0.2489 ± 0.0300 | 0.4808 ± 0.0283 | 0.3267 ± 0.0275 | 0.2026 ± 0.0118 | 0.5726 ± 0.0332 | 75.6 | 6.3 |
| GAT-LSTM | 0.9895 ± 0.0002 | 0.4843 ± 0.0020 | 0.4248 ± 0.0158 | 0.4981 ± 0.0138 | 0.4580 ± 0.0047 | 0.2352 ± 0.0035 | 0.6646 ± 0.0100 | 73.5 | 8.8 |
| **TRACE-X** | 0.9890 ± 0.0007 | **0.4857 ± 0.0386** | 0.4554 ± 0.0603 | 0.4945 ± 0.0182 | **0.4721 ± 0.0362** | **0.2404 ± 0.0066** | **0.6795 ± 0.0186** | 89.4 | 11.3 |

TRACE-X per seed: PR-AUC 0.5391 / 0.4495 / 0.4683.

**Verdict.** TRACE-X has the highest mean PR-AUC, F1, P@K and R@K, and the best single run of any model. However, its advantage over the simpler GAT-LSTM is **within seed noise** (0.486 ± 0.039 vs 0.484 ± 0.002). Both hybrids clearly beat the sequence-only baselines (+0.09 PR-AUC) and the graph-only GNNs (+0.21). Calibration after Platt scaling: TRACE-X Brier 0.00123, ECE 0.0006.

## D. Ablation findings

Paired (same-seed) ΔPR-AUC versus TRACE-X; full table in `reports/results/results_ablation.md`.

**Improved performance:**
- **Gated fusion vs concatenation:** −0.077 when concatenation is used. It is lower on both seeds, so this is the only consistent effect.

**Probably helped (not conclusive):**
- **Combining the graph and sequence branches:** sequence-only −0.037, graph-only −0.013. The full model is better on seed 0 and slightly worse on seed 1.

**Possibly important, not conclusive:**
- **Graph attention vs mean aggregation:** −0.234, but this single run stopped improving after epoch 1, so it may reflect an optimisation failure.

**No measurable effect within noise:**
- learnable time encoding (+0.003);
- graph edge features (+0.028, 1 seed);
- the sender branch (−0.011) and the receiver branch (−0.009) individually;
- sequence length 5 or 20 vs 10 (+0.015 / +0.001).

## E. Explainability findings

All explanations are for TRACE-X seed 0 on 10 test cases (5 TP, 2 FP, 2 FN, 1 TN). Units are logits. Example: `reports/explanations/cases/4723765.md`, an ACH transfer of $7,324 flagged at p = 0.9999.

- **Its top SHAP drivers:** payment_format +4.44, log_usd_paid +1.26, day_of_week +0.84.
- **IG:** the sender's transaction 31 minutes earlier contributes +0.31.

**Quality evaluation:**

| Method | Delete top-k vs random-k (logit drop) | Sparsity | Stability across seeds (Spearman) | Sanity check |
|---|---|---|---|---|
| Kernel SHAP | 9.94 vs 0.68 | top-5 carry 83 % | 0.68 | additivity gap 2e-7 |
| Integrated Gradients | 0.89 vs 0.29 | top-3 carry 41 % | 0.68 | completeness \|Δ\| 0.001 |
| GNNExplainer | −0.03 vs 0.01 | 0 % of masks > 0.5 | 0.996 (an artefact of near-uniform masks) | — |

**Conclusions:**

- SHAP and IG are faithful by deletion tests and moderately stable across seeds.
- GNNExplainer does **not** produce an informative sparse graph explanation for this model. Reports now flag this automatically.
- `day_of_week` contributes in several alerts, which is a possible simulator shortcut.

**Uncertainty (TRACE-X seed 0):**

- **Conformal triage** (α₀ = 0.01, α₁ = 0.1):
  - coverage for normal transactions is 0.974, below its 0.99 target, which indicates drift;
  - coverage for laundering is 0.943;
  - 4.35 % of traffic goes to human review.
- **MC-dropout** standard deviation detects errors with AUROC 0.813.

## F. Testing

- `python -m pytest tests -q` → **42 executed, 42 passed, 0 failed** (18 Nov 2026).
  - Leakage: 12 tests. Models: 21. Training: 3. Explainability: 6.
- Fixed in this session:
  - One stale assertion: the GNNExplainer fidelity test expected probability output, but the code now returns logit units.
  - One autograd warning in the trainer.
- **Remaining issues:**
  - Third-party deprecation warnings (torch.jit and an anyio pytest-plugin alias).
  - Tests use dev-mode artefacts, so a fresh clone must run the dev pipeline and a 1-epoch dev checkpoint first (`scripts/reproduce.sh` does this).

## G. Dashboard status

All six pages were rendered on the final checkpoints and captured with Playwright and Chrome. Screenshots are in `reports/dashboard_screenshots/`:

1. Overview
2. Transaction Investigation
3. Network Explorer
4. Explainable AI
5. Model Laboratory
6. Investigation Reports, with a live-generated report for transaction 4723792 (a false positive), including the diffuse-graph caveat and the disclaimer.

## H. Limitations

- **Dataset and synthetic nature.** HI-Small is simulated, so results measure how well a model learns the simulator's typologies. The test period ends with days of almost exclusively laundering transactions, an artefact that a model may exploit (see `day_of_week`).
- **Class imbalance (0.10 %).** Even the best model has precision of about 0.46 at its operating threshold and P@0.5 % of 0.24, so most alerts still need human review.
- **Compute.** Training was CPU-only:
  - ablations have only 1–2 seeds;
  - there was no hyper-parameter search;
  - TRACE-X has high seed variance, with 2 of 3 seeds stopping at epochs 3–4;
  - training-time measurements are confounded by two jobs running in parallel.
- **Temporal generalisation.** There is a single chronological split, and conformal coverage for normal traffic fell below its target on test.
- **Explainability.**
  - SHAP holds the histories and neighbourhood fixed.
  - IG depends on the baseline.
  - GNNExplainer masks are uninformative for this model.
  - Stability across seeds is only moderate.
  - None of the methods establishes causation or criminal intent.
- **Deployment.** The model has not been validated on real bank data and has no model-risk governance, privacy review or drift monitoring. It is decision support only.

## I. Reproduction

```bash
pip install -r requirements.txt
# dataset: automatic via kagglehub, or put HI-Small_Trans.csv in data/raw/ (or set TRACEX_RAW_FILE)
python -m src.data.pipeline --mode dev
python -m src.data.pipeline --mode full
python -m src.data.profiling --mode full
python -m src.training.run_experiment --exp tracex --seed 0 --mode dev --max-epochs 1 --neg-per-epoch 5000
python -m pytest tests -q
python scripts/run_queue.py --exps tracex,lstm,gat,gru,gcn,graphsage,gat_lstm --seeds 0,1,2
python scripts/run_queue.py --seed-major --seeds 0 --exps abl_sequence_only,abl_graph_only,abl_no_temporal_enc,abl_no_edge_feats,abl_mean_aggr,abl_no_sender,abl_no_receiver,abl_concat_fusion,abl_seq5,abl_seq20
python scripts/run_queue.py --seeds 1 --exps abl_sequence_only,abl_graph_only,abl_concat_fusion,abl_no_temporal_enc
python -m src.evaluation.comparison --mode full
python scripts/explain.py --exp tracex --seed 0 --stability-seeds 1,2
python scripts/uncertainty.py --exp tracex --seed 0
python scripts/benchmark_efficiency.py
streamlit run dashboard/app.py
```
