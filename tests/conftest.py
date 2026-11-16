import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.utils.config import load_yaml, processed_dir  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    """Small synthetic transaction set with many timestamp ties and self-loops."""
    rng = np.random.default_rng(0)
    n, n_acc = 600, 25
    ts = np.sort(rng.integers(0, 40, n))
    src = rng.integers(0, n_acc, n)
    dst = rng.integers(0, n_acc, n)
    dst[::17] = src[::17]  # some self-loops
    usd = rng.lognormal(5, 2, n)
    return {"src": src, "dst": dst, "ts": ts, "usd": usd, "n": n}


@pytest.fixture(scope="session")
def store():
    from src.graph.temporal_graph import TemporalGraphStore

    d = processed_dir(load_yaml("data"), "dev")
    if not (d / "DONE").exists():
        pytest.skip("dev artefacts not built; run python -m src.data.pipeline --mode dev")
    return TemporalGraphStore(d)
