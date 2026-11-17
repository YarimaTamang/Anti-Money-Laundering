# TRACE-X methodology

This document gives the full technical detail behind the README summary.

## 1. Event-time protocol

* HI-Small timestamps have **minute resolution**: 5,078,345 transactions share only 15,018 distinct
  timestamps, and 99.7% of transactions share their minute with at least one other transaction.
  The order of rows inside a minute is not a real event order.
* **Rule:** every input used to score transaction *e* with timestamp *t* is computed only from
  transactions with timestamp **strictly smaller than t**. Transactions in the same minute are
  treated as simultaneous and never see each other.
* Implementation (`src/graph/causal_index.py`): rows are sorted by (account, timestamp). For each row
  we compute `gs` (first row of the account group) and `bs` (first row of its (account, timestamp)
  block). The half-open range `[gs, bs)` is exactly the set of strictly-earlier rows of the same
  account. Counts, volumes, unique counterparties, recency, the history windows and the temporal
  neighbour lists are all exclusive prefix sums or windows over this range. This is fully vectorised:
  full-dataset preprocessing takes about 3 minutes on a laptop CPU.
* Brute-force reference implementations in `tests/test_leakage.py` check the vectorised code on
  synthetic data with heavy timestamp ties.

## 2. Data splits

Chronological by transaction count, with boundaries moved forward to the next timestamp change so
that no minute straddles two splits:

| Split | Share | Purpose |
|---|---|---|
| train | first 60% | gradient updates; FX-rate estimation; feature standardisation |
| val_select | next 10% | early stopping, model selection, F1-optimal decision threshold |
| val_cal | next 10% | Platt scaling, conformal calibration, MC-dropout uncertainty threshold |
| test | last 20% | evaluated once, after all choices are fixed |

## 3. Features

**Account identity.** `bank_code + "_" + account_hex`. Account hex strings alone collide: 515,080
distinct hex strings versus 515,088 distinct (bank, account) pairs.

**Currency normalisation.** Amounts are given in 15 currencies. Units-per-USD rates are estimated
as medians over training cross-currency transactions (for example EUR 0.853, JPY 105.4, BTC 8.4e-5).

**Edge (transaction) features, 27 columns.** log1p(USD amount), log1p(original amount), payment
format one-hot (7), payment currency one-hot (15), cross-bank flag, currency-conversion flag,
self-loop flag. The two continuous columns are standardised with training statistics.

**Node (account-state) features, 11 columns, computed strictly before the transaction.**
log counts of outgoing and incoming transactions, log USD volumes, log mean amounts, log unique
counterparties (outgoing and incoming), log transactions per hour since first appearance, log
minutes since last activity, and a has-history flag.

**Sequences.** The sender's last *L* outgoing transactions and the receiver's last *L* incoming
transactions (L = 10 by default; 5 and 20 in the ablations). Each step has 28 features: the 27
edge features plus a flag for whether the step's counterparty equals the target's counterparty
(repeated transfers). Each step also carries two time deltas: minutes to the target and minutes
to the previous step.

**Temporal neighbourhood.** For every (transaction, endpoint) pair the 10 most recent incident
transactions, in either direction, strictly before the transaction are stored as codes
`2·eid + side_of_counterparty`. Because the neighbour list of a neighbour transaction e' is in turn
causal with respect to t(e'), the codes can be expanded recursively into a bounded 2-hop computation
subgraph: up to 10 hop-1 and 10×5 hop-2 neighbours per endpoint, with no future information at any
depth.

## 4. TRACE-X architecture

Let *z* = MLP(edge features of the target transaction).

**Time encoding** (following the functional time encoding of TGAT, Xu et al. 2020):
Φ(Δt) = cos(ω Δt + φ), with ω, φ ∈ R^16 learnable. ω is initialised geometrically from 1 down to
1e-5 per minute.

**Edge-aware temporal graph attention** (layer *l*, head *h*, message from neighbour *j* to
account *i* over a past transaction with attributes a_ij and lag Δt_ij):

```
u_ij      = [a_ij ‖ dir_ij ‖ Φ(Δt_ij)]
e_ij^h    = a_hᵀ LeakyReLU( W_q^h x_i + W_k^h x_j + W_e^h u_ij )      (GATv2-style scoring)
α_ij^h    = softmax_j(e_ij^h)
m_i       = ‖_h Σ_j α_ij^h ( W_v^h x_j + W_u^h u_ij )
x_i'      = LayerNorm( x_i + ELU( W_s x_i + m_i ) )
```

There are two layers over the 2-hop tree. Graph representation:
r_g = ReLU(W [x'_sender ‖ x'_receiver]).
Layer 2 only propagates hop-1→centre edges; a unit test shows the outputs are identical.

**Sender and receiver LSTMs** use separate weights. Step input = [step features ‖ Φ(Δt_target) ‖ Φ(Δt_prev)].

```
i_t = σ(W_i x_t + U_i h_{t-1} + b_i)      f_t = σ(W_f x_t + U_f h_{t-1} + b_f)
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)      c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t           h_t = o_t ⊙ tanh(c_t)
```

r_s and r_r are the final hidden states, using packed sequences; empty histories give zero vectors.

**Temporal branch.**
r_t = MLP([Φ(Δt since sender's last activity) ‖ Φ(Δt since receiver's last activity) ‖ has-history flags ‖ sin/cos time-of-day ‖ sin/cos day-of-week]).

**Gated fusion** (element-wise, conditioned on all branches and the transaction embedding):

```
g_k = σ( W_g^k [r_g ‖ r_s ‖ r_r ‖ r_t ‖ z] + b_g^k ),   k ∈ {g, s, r, t}
f   = Σ_k g_k ⊙ tanh(W_k r_k + b_k)
ŷ   = w_2ᵀ Dropout(ReLU(W_1 [f ‖ z] + b_1)) + b_2          (logit; sigmoid only at inference)
```

**What is borrowed and what is new.** Graph attention comes from Veličković et al. (2018), the
dynamic (GATv2) scoring from Brody et al. (2022), and the functional time encoding from TGAT
(Xu et al., 2020). The combination is this project's design and is **not** a reproduction of
any published architecture: edge-aware temporal attention over a causal 2-hop neighbourhood,
direction-specific sender and receiver LSTMs, an explicit temporal branch, and element-wise gated
fusion conditioned on the transaction embedding.

## 5. Baselines

All baselines use the same splits, inputs available at prediction time, target transaction
embedding *z*, hidden size (64), optimiser, loss, epoch budget and early-stopping rule.

* **LSTM / GRU:** one shared recurrent encoder over the sender and receiver histories.
  Δt is given as a log-scaled scalar; there is no graph.
* **GCN / GraphSAGE / GAT:** two PyG layers (`GCNConv`, `SAGEConv` with mean aggregation, and
  `GATConv` with `edge_dim`) over the *same* causal 2-hop subgraph. GCN and SAGE cannot use edge
  attributes. GAT receives [edge features ‖ direction ‖ log Δt].
* **GAT-LSTM:** the GAT encoder plus a shared LSTM, with plain concatenation. It has no gating,
  no temporal branch and no learnable time encoding.

## 6. Training

* AdamW (lr 1e-3, weight decay 1e-4), gradient clipping at 1.0, ReduceLROnPlateau
  (factor 0.5, patience 1) on validation PR-AUC.
* Early stopping with patience 3 and at most 20 epochs. The best-validation checkpoint is restored.
* Each epoch uses all 2,299 training positives plus 150,000 freshly sampled training negatives.
  The loss is BCE-with-logits with `pos_weight = n_neg / n_pos` of the epoch sample (about 65).
  Focal loss is available as an optional experiment.
* Validation PR-AUC per epoch is computed on all val_select positives plus a fixed 50k-negative
  sample, with negatives weighted by the inverse sampling rate (Horvitz–Thompson). A unit test
  checks that this reproduces full-population precision, recall and PR-AUC.
* Mixed precision is enabled automatically only on CUDA. This laptop has no CUDA GPU (AMD Radeon),
  and the Ryzen 5800H has no native bf16, so CPU autocast would only slow training down.

## 7. Evaluation

* All test metrics use the **complete test split** (1,015,564 transactions, 1,797 laundering).
* Precision, recall and F1 use the F1-optimal threshold chosen on val_select.
* Alert budget: rank all test transactions by score and take the top K = ⌈b·N_test⌉.
  Precision@K = hits / K; Recall@K = hits / (all test positives), which is the detection rate.
* Calibration: Brier score, ECE (15 uniform bins) and a quantile-binned reliability diagram,
  for both raw probabilities and Platt-scaled probabilities (fitted on val_cal).
* Efficiency: trainable parameters, wall-clock training time, test throughput, single-transaction
  latency (median of 50, including causal batch construction) and peak process working set.

## 8. Explainability

| Method | What is explained | Implementation notes |
|---|---|---|
| GNNExplainer (adapted) | Which past transactions (edges) in the 2-hop subgraph, and which edge/node feature columns, the graph branch relies on | Ying et al. objective (log-likelihood of the predicted class + size + entropy penalties); soft masks multiply messages inside `EdgeTemporalGATConv` |
| Kernel SHAP | 31 interpretable features: target transaction attributes (categoricals as single features), sender and receiver account state, time of day and week | `shap.KernelExplainer`; background = 12 random training transactions; history and neighbourhood held fixed; additivity verified |
| Integrated Gradients | Each historical transaction in the sender and receiver sequences, plus per-feature totals | 64-step trapezoid rule; baseline = 0 in standardised space; completeness verified per case |

Quality evaluation (`src/explainability/fidelity.py`):

* **Fidelity.** Delete the top-k elements (edges, features, history steps), measure the change in
  probability, and compare with deleting k random elements. GNNExplainer also reports fid⁻
  (keep only the top-k edges).
* **Sparsity.** Fraction of edges with mask > 0.5; share of |attribution| carried by the top 3/5 elements.
* **Stability.** Spearman correlation and top-k Jaccard overlap between explanations of the same
  transactions from models trained with different seeds.

## 9. Uncertainty

* **Class-conditional split conformal** (calibrated on val_cal) maps each test transaction to one
  of three outcomes: {normal}, {suspicious}, or {both or neither} → human review. Coverage
  guarantees hold per class under exchangeability. Chronological drift can violate exchangeability,
  so the coverage actually achieved on test is reported.
* **MC Dropout** (T = 20) gives a predictive standard deviation. Review threshold = 95th percentile
  on val_cal. Evaluated on an IPW-weighted test subsample because T full test passes are too
  expensive on CPU.
