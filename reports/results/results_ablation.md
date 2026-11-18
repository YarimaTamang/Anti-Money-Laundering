# Ablation study (full test split, mean ± std over completed seeds)

| Model | Seeds | pr_auc | roc_auc | f1 | r@0.5% | p@0.5% | params | train_min |
|---|---|---|---|---|---|---|---|---|
| TRACE-X (proposed) | 3 | 0.4857 ± 0.0386 | 0.9890 ± 0.0007 | 0.4721 ± 0.0362 | 0.6795 ± 0.0186 | 0.2404 ± 0.0066 | 238369 ± 0 | 89.4 ± 59.1 |
| Sequence branches only | 2 | 0.4573 ± 0.0015 | 0.9883 ± 0.0003 | 0.4693 ± 0.0011 | 0.6297 ± 0.0075 | 0.2228 ± 0.0027 | 111873 ± 0 | 8.4 ± 2.3 |
| Graph branch only | 2 | 0.4812 ± 0.0200 | 0.9855 ± 0.0006 | 0.4747 ± 0.0261 | 0.6728 ± 0.0017 | 0.2381 ± 0.0006 | 80705 ± 0 | 80.3 ± 1.8 |
| w/o temporal encoding | 2 | 0.4970 ± 0.0162 | 0.9895 ± 0.0002 | 0.4698 ± 0.0214 | 0.6664 ± 0.0003 | 0.2358 ± 0.0001 | 175361 ± 0 | 35.7 ± 1.1 |
| w/o edge features (graph) | 1 | 0.5669 | 0.9904 | 0.5470 | 0.7034 | 0.2489 | 231201 | 93.7 |
| w/o graph attention (mean aggr.) | 1 | 0.3049 | 0.9858 | 0.3268 | 0.5537 | 0.1959 | 216097 | 10.6 |
| w/o sender history | 1 | 0.5285 | 0.9899 | 0.5107 | 0.7140 | 0.2527 | 169089 | 59.2 |
| w/o receiver history | 1 | 0.5297 | 0.9902 | 0.5052 | 0.6989 | 0.2473 | 169089 | 43.9 |
| concat instead of gated fusion | 2 | 0.4175 ± 0.0052 | 0.9888 ± 0.0001 | 0.4299 ± 0.0030 | 0.6784 ± 0.0033 | 0.2401 ± 0.0012 | 151841 ± 0 | 44.4 ± 21.6 |
| sequence length 5 | 1 | 0.5538 | 0.9901 | 0.5107 | 0.7257 | 0.2568 | 238369 | 44.4 |
| sequence length 20 | 1 | 0.5404 | 0.9902 | 0.5180 | 0.7245 | 0.2564 | 238369 | 62.8 |

ΔPR-AUC relative to full TRACE-X. 'vs mean' compares with the 3-seed TRACE-X mean; 'paired' compares each ablation seed with the TRACE-X run of the same seed and averages over the seeds both share. TRACE-X itself varies across seeds (see per-seed values below), so differences smaller than that spread are not evidence of a real effect.

| Variant | Seeds | ΔPR-AUC vs mean | ΔPR-AUC paired (same seeds) |
|---|---|---|---|
| Sequence branches only | 2 | -0.0284 | -0.0370 |
| Graph branch only | 2 | -0.0044 | -0.0131 |
| w/o temporal encoding | 2 | +0.0113 | +0.0027 |
| w/o edge features (graph) | 1 | +0.0813 | +0.0278 |
| w/o graph attention (mean aggr.) | 1 | -0.1808 | -0.2343 |
| w/o sender history | 1 | +0.0428 | -0.0106 |
| w/o receiver history | 1 | +0.0441 | -0.0094 |
| concat instead of gated fusion | 2 | -0.0682 | -0.0769 |
| sequence length 5 | 1 | +0.0681 | +0.0146 |
| sequence length 20 | 1 | +0.0548 | +0.0013 |

TRACE-X per-seed test PR-AUC: seed 0: 0.5391, seed 1: 0.4495, seed 2: 0.4683