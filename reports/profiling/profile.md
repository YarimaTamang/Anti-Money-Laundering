# Dataset profile — IBM AML HI-Small (full mode)

| Statistic | Value |
|---|---|
| Transactions | 5,078,345 |
| Unique sender accounts | 496,999 |
| Unique receiver accounts | 420,640 |
| Unique accounts (bank+account) | 515,088 |
| Unique banks | 30,528 |
| Missing values (raw) | 0 |
| Exact duplicate rows (raw, kept) | 9 |
| Laundering transactions | 5,177 (0.102%) |
| Time range | 2022-09-01 00:00:00 → 2022-09-18 16:18:00 |
| Distinct timestamps (minute resolution) | 15,018 |
| Cross-bank transactions | 86.4% |
| Currency conversions | 1.42% |
| Self-loop transactions | 11.6% |
| Median amount (USD) normal / laundering | 891 / 5,879 |
| Unique directed account pairs | 1,015,736 |
| Reciprocal pair fraction | 0.3694 |
| Weakly connected components | 114,139 (largest 372,089) |
| Max out / in degree | 168,672 / 1,084 |

## Payment formats

| Format | Count | Laundering rate |
|---|---|---|
| Cheque | 1,864,331 | 0.017% |
| Credit Card | 1,323,324 | 0.016% |
| ACH | 600,797 | 0.746% |
| Cash | 490,891 | 0.022% |
| Reinvestment | 481,056 | 0.000% |
| Wire | 171,855 | 0.000% |
| Bitcoin | 146,091 | 0.038% |

## Transactions per day

| Day | Transactions | Laundering |
|---|---|---|
| 2022-09-01 | 1,114,921 | 322 |
| 2022-09-02 | 754,449 | 408 |
| 2022-09-03 | 207,382 | 391 |
| 2022-09-04 | 207,430 | 407 |
| 2022-09-05 | 482,650 | 471 |
| 2022-09-06 | 482,089 | 531 |
| 2022-09-07 | 482,751 | 497 |
| 2022-09-08 | 482,773 | 539 |
| 2022-09-09 | 654,467 | 514 |
| 2022-09-10 | 208,325 | 442 |
| 2022-09-11 | 396 | 232 |
| 2022-09-12 | 281 | 170 |
| 2022-09-13 | 184 | 106 |
| 2022-09-14 | 121 | 70 |
| 2022-09-15 | 46 | 28 |
| 2022-09-16 | 46 | 26 |
| 2022-09-17 | 23 | 15 |
| 2022-09-18 | 11 | 8 |

## Chronological split

| Split | Transactions | Laundering | Time range |
|---|---|---|---|
| train | 3,047,216 | 2,299 | 2022-09-01T00:00 → 2022-09-06T13:36 |
| val_select | 508,115 | 558 | 2022-09-06T13:37 → 2022-09-07T14:56 |
| val_cal | 507,450 | 523 | 2022-09-07T14:57 → 2022-09-08T16:12 |
| test | 1,015,564 | 1,797 | 2022-09-08T16:13 → 2022-09-18T16:18 |

Figures: `class_distribution.png`, `amount_distribution.png`, `frequency_distribution.png`, `temporal_distribution.png`, `payment_format_distribution.png`, `currency_distribution.png`, `degree_distribution.png`