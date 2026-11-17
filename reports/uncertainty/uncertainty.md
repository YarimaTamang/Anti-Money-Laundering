# Uncertainty & human review (TRACE-X, seed 0)

## Split conformal (class-conditional), calibrated on val_cal, full test split

α₀ bounds the share of normal transactions pushed out of the 'normal' set; α₁ bounds the share of laundering transactions pushed out of the 'suspicious' set. Values are fixed policy choices, not tuned on test labels.

| α₀ (normal) / α₁ (laundering) | coverage normal | coverage laundering | confident normal | confident suspicious | human review | laundering sent to review | retained precision | retained recall |
|---|---|---|---|---|---|---|---|---|
| 0.1/0.1 | 0.8896 | 0.9432 | 0.8880 | 0.07133 | 0.0406 | 0.0323 | 0.0234 | 0.9747 |
| 0.05/0.05 | 0.9336 | 0.9716 | 0.8917 | 0.06793 | 0.0403 | 0.0284 | 0.0246 | 0.9708 |
| 0.02/0.1 | 0.9579 | 0.9432 | 0.9287 | 0.04361 | 0.0277 | 0.0223 | 0.0374 | 0.9419 |
| 0.01/0.1 | 0.9737 | 0.9432 | 0.9287 | 0.02783 | 0.0435 | 0.0428 | 0.0573 | 0.9407 |
| 0.01/0.05 | 0.9737 | 0.9716 | 0.8917 | 0.02783 | 0.0805 | 0.0712 | 0.0573 | 0.9694 |
| 0.005/0.1 | 0.9903 | 0.9432 | 0.9287 | 0.01118 | 0.0602 | 0.1024 | 0.1331 | 0.9368 |

Deployed policy α₀=0.01, α₁=0.1: p ≤ 0.7375 → 'normal' in set; p ≥ 0.5942 → 'suspicious' in set; both or neither → human review.

## MC Dropout (T=20), test subsample (all positives + uniform negatives, IPW-weighted)

- Review rule: std > 0.0234 (95th percentile on val_cal)
- Abstention rate: 0.0801; laundering transactions routed to review: 0.0812
- Precision/recall on retained predictions: 0.5300 / 0.4985 (all: 0.5300 / 0.4580)
- AUROC of MC std for detecting misclassifications: 0.8129849624266205

_The system is a decision-support tool; an alert or a 'confident' outcome does not establish that money laundering occurred._