#!/usr/bin/env bash
# Full TRACE-X reproduction (CPU). Long-running: see README for measured run times.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -r requirements.txt

# 1. data: kagglehub download (or place HI-Small_Trans.csv in data/raw) + cached preprocessing
python -m src.data.pipeline --mode dev      # small chronological prefix for smoke tests
python -m src.data.pipeline --mode full
python -m src.data.profiling --mode full

# 2. tests (dev artefacts + a 1-epoch dev checkpoint are needed by some tests)
python -m src.training.run_experiment --exp tracex --seed 0 --mode dev --max-epochs 1 --neg-per-epoch 5000
python -m pytest tests -q

# 3. training: proposed model + baselines (3 seeds); ablations: seed 0 for all, seed 1 for the
#    four key ablations (compute budget on a CPU-only laptop; use --seeds 0,1,2 for the full design)
python scripts/run_queue.py --exps tracex,lstm,gat,gru,gcn,graphsage,gat_lstm --seeds 0,1,2
python scripts/run_queue.py --seed-major --seeds 0 --exps \
  abl_sequence_only,abl_graph_only,abl_no_temporal_enc,abl_no_edge_feats,abl_mean_aggr,abl_no_sender,abl_no_receiver,abl_concat_fusion,abl_seq5,abl_seq20
python scripts/run_queue.py --seeds 1 --exps abl_sequence_only,abl_graph_only,abl_concat_fusion,abl_no_temporal_enc

# 4. evaluation tables + figures
python -m src.evaluation.comparison --mode full

# 5. explanations, uncertainty
python scripts/explain.py --exp tracex --seed 0 --stability-seeds 1,2
python scripts/uncertainty.py --exp tracex --seed 0
python scripts/benchmark_efficiency.py        # controlled efficiency benchmark (run on an idle machine)

# 6. dashboard / API
streamlit run dashboard/app.py
# uvicorn src.api:app --port 8000
