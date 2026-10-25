from pathlib import Path
import sys
import numpy as np
import pandas as pd

STAGE2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE2 / "src"))
from engine import solve_weights, metrics, block_bootstrap_indices, ASSETS, MAX_WEIGHT

def test_optimizer_weights_obey_constraints():
    mu=np.array([.10,.08,.12,.07,.11,.02])
    cov=np.eye(6)*.04
    for objective in ["max_sharpe","min_variance"]:
        w=solve_weights(mu,cov,objective)
        assert np.isfinite(w).all()
        assert abs(w.sum()-1)<1e-6
        assert w.min()>=-1e-7
        assert w.max()<=MAX_WEIGHT+1e-5

def test_metrics_known_simple_returns():
    x=np.array([.01,.01,-.005,.01,-.005,.01])
    result=metrics(x,"test")
    assert result["Observations"]==len(x)
    assert np.isfinite(result["Total Return"])
    assert result["Maximum Drawdown"]<=0

def test_block_bootstrap_indices_length_and_range():
    idx=block_bootstrap_indices(100,10,np.random.default_rng(42))
    assert len(idx)==100
    assert idx.min()>=0 and idx.max()<100

def test_input_file_exists_from_project_layout():
    project_root=STAGE2.parents[0]
    assert (project_root/"data"/"processed"/"daily_returns.csv").exists()
