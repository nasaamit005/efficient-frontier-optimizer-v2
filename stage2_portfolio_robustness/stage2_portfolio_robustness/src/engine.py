from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_ROOT = Path(__file__).resolve().parents[1]
INPUT_RETURNS = PROJECT_ROOT / "data" / "processed" / "daily_returns.csv"
INPUT_FRONTIER_WEIGHTS = PROJECT_ROOT / "data" / "processed" / "frontier_weights.csv"
OUT = STAGE2_ROOT / "data" / "processed"
REPORTS = STAGE2_ROOT / "reports"
DETAILS = REPORTS / "analysis_details"
DASH = STAGE2_ROOT / "dashboard"
DEPLOY = STAGE2_ROOT / "github_deploy"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
MAX_WEIGHT = 0.40
ROLLING_WINDOW = 756
REBALANCE_STEP = 21
BOOTSTRAP_REPS = 60
BOOTSTRAP_BLOCK = 21
FRONTIER_POINTS = 20
SEED = 42
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BND"]
BENCHMARK = "^GSPC"

COLORS = {
    "navy": "#17324D", "blue": "#2878B5", "teal": "#159D91",
    "orange": "#E58A2B", "red": "#D94F5C", "purple": "#8064A2",
    "green": "#4F956A", "grey": "#7B8794", "light": "#F4F7FB",
    "grid": "#DCE4EE", "ink": "#263445"
}
TEMPLATE = "plotly_white"


def ensure_dirs():
    for p in (OUT, REPORTS, DETAILS, DASH, DEPLOY):
        p.mkdir(parents=True, exist_ok=True)


def load_data():
    if not INPUT_RETURNS.exists():
        raise FileNotFoundError(f"Required source data not found: {INPUT_RETURNS}")
    df = pd.read_csv(INPUT_RETURNS, parse_dates=["Date"]).sort_values("Date")
    if "Date" not in df or not set(ASSETS).issubset(df.columns):
        raise ValueError(f"daily_returns.csv must contain Date and {ASSETS}")
    df = df[["Date"] + ASSETS + ([BENCHMARK] if BENCHMARK in df.columns else [])]
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=ASSETS).reset_index(drop=True)
    if (df[ASSETS] <= -1).any().any():
        raise ValueError("Returns <= -100% found; compound returns are invalid.")
    if len(df) < 900:
        raise ValueError("At least 900 complete observations are required for Stage 2.")
    return df


def solve_weights(mu, cov, objective="max_sharpe", max_weight=MAX_WEIGHT):
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    if cov.shape != (n, n) or not np.isfinite(mu).all() or not np.isfinite(cov).all():
        raise ValueError("Invalid mean/covariance dimensions or non-finite values.")
    cov = (cov + cov.T) / 2
    eigmin = float(np.linalg.eigvalsh(cov).min())
    if eigmin < 0:
        cov += np.eye(n) * (abs(eigmin) + 1e-10)
    bounds = [(0.0, max_weight)] * n
    if n * max_weight < 1:
        raise ValueError("Weight cap makes a fully invested portfolio infeasible.")
    x0 = np.repeat(1/n, n)
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    def vol(w):
        return float(np.sqrt(max(w @ cov @ w, 1e-16)))
    if objective == "min_variance":
        fun = lambda w: float(w @ cov @ w)
    elif objective == "max_sharpe":
        fun = lambda w: -float((w @ mu - RISK_FREE_RATE) / vol(w))
    else:
        raise ValueError(f"Unknown objective: {objective}")
    res = minimize(fun, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1500, "ftol": 1e-11})
    if not res.success:
        # Multiple deterministic starts reduce avoidable solver failures.
        starts = [np.repeat(1/n, n), np.full(n, 1/n)]
        for start in starts:
            res = minimize(fun, start, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"maxiter": 3000, "ftol": 1e-12})
            if res.success:
                break
    if not res.success:
        raise RuntimeError(f"{objective} optimization failed: {res.message}")
    w = np.clip(res.x, 0, max_weight)
    w /= w.sum()
    if abs(w.sum()-1) > 1e-6 or w.max() > max_weight + 1e-5 or w.min() < -1e-7:
        raise RuntimeError("Optimizer returned weights outside constraints.")
    return w


def metrics(returns, label="Portfolio"):
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(x) < 2:
        return {"Strategy": label, "Observations": len(x)}
    wealth = np.cumprod(1 + x)
    years = len(x) / TRADING_DAYS
    cagr = wealth[-1] ** (1/years) - 1 if wealth[-1] > 0 else np.nan
    vol = np.std(x, ddof=1) * np.sqrt(TRADING_DAYS)
    ann_mean = np.mean(x) * TRADING_DAYS
    downside = np.minimum(x, 0)
    dd = wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1
    mdd = float(np.min(dd))
    downside_dev = np.sqrt(np.mean(downside**2) * TRADING_DAYS)
    return {
        "Strategy": label, "Observations": int(len(x)), "Total Return": float(wealth[-1]-1),
        "CAGR": float(cagr), "Annualized Mean Return": float(ann_mean),
        "Annualized Volatility": float(vol),
        "Sharpe Ratio": float((ann_mean-RISK_FREE_RATE)/vol) if vol > 1e-12 else np.nan,
        "Sortino Ratio": float((ann_mean-RISK_FREE_RATE)/downside_dev) if downside_dev > 1e-12 else np.nan,
        "Maximum Drawdown": mdd, "Calmar Ratio": float(cagr/abs(mdd)) if mdd < -1e-12 else np.nan
    }


def block_bootstrap_indices(n, block_len, rng):
    idx = []
    while len(idx) < n:
        start = int(rng.integers(0, n))
        idx.extend((start + np.arange(block_len)) % n)
    return np.asarray(idx[:n], dtype=int)


def save_csv(name, df):
    path = OUT / name
    df.to_csv(path, index=False)
    return path


def style_fig(fig, title, subtitle=None):
    fig.update_layout(
        template=TEMPLATE, title={"text": title + (f"<br><sup>{subtitle}</sup>" if subtitle else ""),
                                  "x": 0.02, "xanchor": "left"},
        paper_bgcolor="white", plot_bgcolor="white", font={"family": "Arial", "color": COLORS["ink"]},
        margin={"l": 60, "r": 35, "t": 90, "b": 60}, legend={"bgcolor": "rgba(255,255,255,0.7)"},
        hovermode="closest"
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


def fig_html(fig, path, include_plotlyjs="cdn"):
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True,
                   config={"responsive": True, "displaylogo": False})


def make_frontier(mu, cov, n_points=FRONTIER_POINTS):
    min_w = solve_weights(mu, cov, "min_variance")
    max_w = solve_weights(mu, cov, "max_sharpe")
    # Generate a feasible return grid between minimum-variance return and max feasible return.
    from scipy.optimize import linprog
    n = len(mu)
    lp = linprog(-mu, A_eq=np.ones((1,n)), b_eq=[1.0],
                 bounds=[(0, MAX_WEIGHT)]*n, method="highs")
    if not lp.success:
        raise RuntimeError("Could not find feasible maximum-return portfolio.")
    low, high = float(min_w @ mu), float(lp.x @ mu)
    targets = np.linspace(low, high, n_points)
    rows = []
    for target in targets:
        cons = [{"type":"eq","fun":lambda w: np.sum(w)-1},
                {"type":"eq","fun":lambda w, t=target: float(w@mu)-t}]
        res = minimize(lambda w: float(w@cov@w), min_w, method="SLSQP",
                       bounds=[(0, MAX_WEIGHT)]*n, constraints=cons,
                       options={"maxiter":1500,"ftol":1e-10})
        if res.success:
            w = np.clip(res.x,0,MAX_WEIGHT); w /= w.sum()
            vol = float(np.sqrt(max(w@cov@w,0)))
            ret = float(w@mu)
            rows.append({"Target Return": target, "Expected Return": ret,
                         "Volatility": vol, "Sharpe Ratio": (ret-RISK_FREE_RATE)/vol if vol else np.nan})
    return pd.DataFrame(rows), min_w, max_w


def run_all():
    ensure_dirs()
    df = load_data()
    rets = df[ASSETS].copy()
    dates = df["Date"]
    n = len(rets)
    rng = np.random.default_rng(SEED)
    daily_mu = rets.mean().to_numpy()
    sample_cov_daily = rets.cov().to_numpy()
    sample_mu_ann = daily_mu * TRADING_DAYS
    sample_cov_ann = sample_cov_daily * TRADING_DAYS

    assumptions = pd.DataFrame([
        ("Input returns", str(INPUT_RETURNS.relative_to(PROJECT_ROOT)), "Existing processed daily returns; no redownload"),
        ("Assets", ", ".join(ASSETS), "Portfolio universe; S&P 500 used only as a benchmark when available"),
        ("Trading days per year", TRADING_DAYS, "Annualization convention"),
        ("Risk-free rate", RISK_FREE_RATE, "Annual rate used in Sharpe objectives and ratios"),
        ("Maximum asset weight", MAX_WEIGHT, "Long-only individual weight cap"),
        ("Rolling estimation window", ROLLING_WINDOW, "Daily observations; approximately three trading years"),
        ("Rebalance interval", REBALANCE_STEP, "Trading observations"),
        ("Bootstrap replications", BOOTSTRAP_REPS, "Computational baseline; increase for final research"),
        ("Bootstrap block length", BOOTSTRAP_BLOCK, "Moving circular blocks preserve short-run dependence"),
        ("Random seed", SEED, "Reproducibility"),
        ("In-sample fraction", 0.70, "Chronological split; first 70% train, last 30% test"),
        ("Missing data policy", "Drop rows missing any portfolio asset", "Common aligned universe for covariance/optimization"),
        ("Transaction costs", "Not included", "Turnover and cost sensitivity are not a live-trading claim"),
    ], columns=["Assumption", "Value", "Definition"])
    save_csv("assumptions.csv", assumptions)

    # 10: Frontier transition map from the existing optimizer's actual frontier weights.
    transition_rows = []
    if INPUT_FRONTIER_WEIGHTS.exists():
        fw = pd.read_csv(INPUT_FRONTIER_WEIGHTS)
        if {"Frontier Point","Asset","Weight"}.issubset(fw.columns):
            wide = fw.pivot_table(index="Frontier Point", columns="Asset", values="Weight", aggfunc="first").sort_index()
            wide = wide.reindex(columns=ASSETS, fill_value=0).fillna(0)
            for i in range(1, len(wide)):
                prev, cur = wide.iloc[i-1], wide.iloc[i]
                transition_rows.append({
                    "From Frontier Point": wide.index[i-1], "To Frontier Point": wide.index[i],
                    "One_Way_Turnover": float(np.abs(cur-prev).sum()/2),
                    "Largest_Absolute_Weight_Change": float(np.abs(cur-prev).max()),
                    "Largest_Change_Asset": str(np.abs(cur-prev).idxmax()),
                    **{f"Delta_{a}": float(cur[a]-prev[a]) for a in ASSETS}
                })
            trans = pd.DataFrame(transition_rows)
            save_csv("10_frontier_transition_map.csv", trans)
            trans_wide = wide.reset_index()
            save_csv("10_frontier_weights_wide.csv", trans_wide)
        else:
            trans = pd.DataFrame(columns=["From Frontier Point","To Frontier Point","One_Way_Turnover"])
            trans_wide = pd.DataFrame()
    else:
        trans = pd.DataFrame(columns=["From Frontier Point","To Frontier Point","One_Way_Turnover"])
        trans_wide = pd.DataFrame()
    if trans.empty:
        # A transparent unavailable record; do not fabricate transition findings.
        trans = pd.DataFrame([{"Status":"Unavailable: existing frontier_weights.csv missing or incompatible"}])
        save_csv("10_frontier_transition_map.csv", trans)

    # 11: Rolling 3-year weights, each portfolio held over the next 21 observations.
    rolling_weights, rolling_realized, rolling_checks = [], [], []
    start = ROLLING_WINDOW
    if n <= start + REBALANCE_STEP:
        start = max(252, int(n*0.60))
    for formation in range(start, n, REBALANCE_STEP):
        hist = rets.iloc[max(0, formation-ROLLING_WINDOW):formation]
        if len(hist) < 252:
            continue
        mu = hist.mean().to_numpy()*TRADING_DAYS
        cov = hist.cov().to_numpy()*TRADING_DAYS
        try:
            w_max = solve_weights(mu,cov,"max_sharpe")
            w_min = solve_weights(mu,cov,"min_variance")
        except Exception as exc:
            rolling_checks.append({"Formation_Date":dates.iloc[formation].date().isoformat(),
                                   "Check":"optimizer", "Passed":False, "Details":str(exc)})
            continue
        for strategy, w in [("Rolling Maximum Sharpe",w_max),("Rolling Minimum Variance",w_min)]:
            rolling_weights.append({"Formation_Date":dates.iloc[formation].date().isoformat(),
                                    "Strategy":strategy, **dict(zip(ASSETS,w))})
            end = min(n, formation+REBALANCE_STEP)
            for j in range(formation,end):
                rolling_realized.append({"Date":dates.iloc[j].date().isoformat(),"Strategy":strategy,
                                         "Return":float(rets.iloc[j].to_numpy()@w),
                                         "Formation_Date":dates.iloc[formation].date().isoformat()})
            rolling_checks.append({"Formation_Date":dates.iloc[formation].date().isoformat(),
                                   "Check":f"{strategy} weights sum", "Passed":bool(abs(w.sum()-1)<1e-6),
                                   "Details":f"sum={w.sum():.8f}; max={w.max():.6f}"})
    rw = pd.DataFrame(rolling_weights)
    rr = pd.DataFrame(rolling_realized).drop_duplicates(["Date","Strategy"], keep="last")
    if not rw.empty:
        save_csv("11_rolling_portfolio_weights.csv", rw)
        save_csv("11_rolling_realized_returns.csv", rr)
    else:
        save_csv("11_rolling_portfolio_weights.csv", pd.DataFrame(columns=["Formation_Date","Strategy"]+ASSETS))
        save_csv("11_rolling_realized_returns.csv", pd.DataFrame(columns=["Date","Strategy","Return","Formation_Date"]))

    # 12: chronological 70/30 holdout. Estimate weights only on training data.
    split = int(n*0.70)
    train, test = rets.iloc[:split], rets.iloc[split:]
    mu_train = train.mean().to_numpy()*TRADING_DAYS
    cov_train = train.cov().to_numpy()*TRADING_DAYS
    strategy_weights = {
        "Maximum Sharpe (train-only)": solve_weights(mu_train,cov_train,"max_sharpe"),
        "Minimum Variance (train-only)": solve_weights(mu_train,cov_train,"min_variance"),
        "Equal Weight": np.repeat(1/len(ASSETS),len(ASSETS)),
    }
    perf_rows, weight_rows, path_rows = [], [], []
    for name,w in strategy_weights.items():
        weight_rows.extend([{"Strategy":name,"Asset":a,"Weight":float(v)} for a,v in zip(ASSETS,w)])
        for period, part in [("In-Sample",train),("Out-of-Sample",test)]:
            pr = part.to_numpy()@w
            perf_rows.append({"Period":period, **metrics(pr,name)})
            wealth=np.cumprod(1+pr)
            for d,r,wl in zip(df["Date"].iloc[part.index],pr,wealth):
                path_rows.append({"Date":d.date().isoformat(),"Strategy":name,"Period":period,
                                  "Daily Return":float(r),"Growth of $1":float(wl)})
    perf = pd.DataFrame(perf_rows)
    save_csv("12_insample_outofsample_metrics.csv",perf)
    save_csv("12_training_weights.csv",pd.DataFrame(weight_rows))
    save_csv("12_performance_paths.csv",pd.DataFrame(path_rows))

    # 13: trailing 252-observation Sharpe for individual assets; descriptive historical series.
    roll_rows=[]
    for col in ASSETS + ([BENCHMARK] if BENCHMARK in df.columns else []):
        x = pd.to_numeric(df[col],errors="coerce")
        roll_mean=x.rolling(TRADING_DAYS,min_periods=TRADING_DAYS).mean()*TRADING_DAYS
        roll_vol=x.rolling(TRADING_DAYS,min_periods=TRADING_DAYS).std(ddof=1)*np.sqrt(TRADING_DAYS)
        sr=(roll_mean-RISK_FREE_RATE)/roll_vol.replace(0,np.nan)
        for d,val,mn,vol in zip(dates,sr,roll_mean,roll_vol):
            if pd.notna(val):
                roll_rows.append({"Date":d.date().isoformat(),"Asset":col,"Rolling_252d_Sharpe":float(val),
                                  "Rolling_Annualized_Mean":float(mn),"Rolling_Annualized_Volatility":float(vol)})
    roll_sharpe=pd.DataFrame(roll_rows)
    save_csv("13_rolling_252d_sharpe.csv",roll_sharpe)

    # 14/15: aligned circular block bootstrap, portfolio metrics and bootstrap frontier.
    # Use training-period expected returns/covariance as the portfolio-design sample.
    boot_rows, boot_frontier_rows = [], []
    train_arr = train.to_numpy()
    target_grid = None
    for b in range(BOOTSTRAP_REPS):
        idx=block_bootstrap_indices(len(train_arr),BOOTSTRAP_BLOCK,rng)
        sample=train_arr[idx,:]
        mu=sample.mean(axis=0)*TRADING_DAYS
        cov=np.cov(sample,rowvar=False,ddof=1)*TRADING_DAYS
        try:
            w=solve_weights(mu,cov,"max_sharpe")
            vol=float(np.sqrt(max(w@cov@w,0)))
            ret=float(w@mu)
            boot_rows.append({"Replication":b+1,"Expected_Annual_Return":ret,"Annualized_Volatility":vol,
                              "Estimated_Sharpe":(ret-RISK_FREE_RATE)/vol if vol>1e-12 else np.nan,
                              **{f"Weight_{a}":float(v) for a,v in zip(ASSETS,w)}})
            # Efficient frontier with fixed target grid derived from this bootstrap sample.
            fw_min=solve_weights(mu,cov,"min_variance")
            from scipy.optimize import linprog
            lp=linprog(-mu,A_eq=np.ones((1,len(ASSETS))),b_eq=[1],
                       bounds=[(0,MAX_WEIGHT)]*len(ASSETS),method="highs")
            if lp.success:
                lo,hi=float(fw_min@mu),float(lp.x@mu)
                if target_grid is None:
                    target_grid=np.linspace(lo,hi,FRONTIER_POINTS)
                for target_id,target in enumerate(target_grid,1):
                    cons=[{"type":"eq","fun":lambda z:np.sum(z)-1},
                          {"type":"eq","fun":lambda z,t=target:float(z@mu)-t}]
                    res=minimize(lambda z:float(z@cov@z),fw_min,method="SLSQP",
                                 bounds=[(0,MAX_WEIGHT)]*len(ASSETS),constraints=cons,
                                 options={"maxiter":500,"ftol":1e-9})
                    if res.success:
                        ww=np.clip(res.x,0,MAX_WEIGHT); ww/=ww.sum()
                        vv=float(np.sqrt(max(ww@cov@ww,0))); rr0=float(ww@mu)
                        boot_frontier_rows.append({"Replication":b+1,"Frontier_Point":target_id,
                                                   "Target_Return":float(target),"Expected_Return":rr0,
                                                   "Volatility":vv,"Sharpe_Ratio":(rr0-RISK_FREE_RATE)/vv if vv>1e-12 else np.nan})
        except Exception as exc:
            warnings.warn(f"Bootstrap replication {b+1} skipped: {exc}")
    boot = pd.DataFrame(boot_rows)
    bf = pd.DataFrame(boot_frontier_rows)
    save_csv("14_bootstrap_frontier_points.csv",bf)
    save_csv("15_bootstrap_sharpe_distribution.csv",boot)

    # Bootstrap Sharpe distribution on strictly held-out daily returns, block-resampling OOS realized returns.
    oos_w = strategy_weights["Maximum Sharpe (train-only)"]
    oos_port = test.to_numpy()@oos_w
    sharpe_boot=[]
    for b in range(BOOTSTRAP_REPS*5):
        idx=block_bootstrap_indices(len(oos_port),BOOTSTRAP_BLOCK,rng)
        stats=metrics(oos_port[idx],"OOS Maximum Sharpe")
        sharpe_boot.append({"Replication":b+1,"Sharpe_Ratio":stats.get("Sharpe Ratio",np.nan),
                            "CAGR":stats.get("CAGR",np.nan),"Annualized_Volatility":stats.get("Annualized Volatility",np.nan)})
    sboot=pd.DataFrame(sharpe_boot)
    save_csv("15_oos_sharpe_block_bootstrap.csv",sboot)

    # 16: sample vs Ledoit-Wolf covariance, with same training means.
    lw = LedoitWolf().fit(train.to_numpy())
    cov_lw = lw.covariance_ * TRADING_DAYS
    cov_sample = train.cov().to_numpy()*TRADING_DAYS
    cov_rows=[]
    for i,a in enumerate(ASSETS):
        for j,b in enumerate(ASSETS):
            cov_rows.append({"Asset_1":a,"Asset_2":b,"Sample_Covariance_Annualized":float(cov_sample[i,j]),
                             "Ledoit_Wolf_Covariance_Annualized":float(cov_lw[i,j]),
                             "Sample_Correlation":float(train[a].corr(train[b])),
                             "Ledoit_Wolf_Correlation":float(cov_lw[i,j]/np.sqrt(cov_lw[i,i]*cov_lw[j,j]))})
    save_csv("16_covariance_comparison.csv",pd.DataFrame(cov_rows))
    cov_method_rows=[]
    for method,cov in [("Sample covariance",cov_sample),("Ledoit-Wolf covariance",cov_lw)]:
        for obj in ["max_sharpe","min_variance"]:
            w=solve_weights(mu_train,cov,obj)
            train_port=train.to_numpy()@w
            test_port=test.to_numpy()@w
            row={"Covariance_Estimator":method,"Objective":obj,"Expected_Annual_Return":float(w@mu_train),
                 "Estimated_Annualized_Volatility":float(np.sqrt(max(w@cov@w,0))),
                 "Estimated_Sharpe":float((w@mu_train-RISK_FREE_RATE)/np.sqrt(max(w@cov@w,1e-16)))}
            row.update({f"Weight_{a}":float(v) for a,v in zip(ASSETS,w)})
            row.update({f"Test_{k}":v for k,v in metrics(test_port,obj).items() if k != "Strategy"})
            cov_method_rows.append(row)
    cov_methods=pd.DataFrame(cov_method_rows)
    save_csv("16_covariance_optimizer_comparison.csv",cov_methods)

    # Data quality / validation records.
    checks=[]
    def check(name,passed,detail):
        checks.append({"Check":name,"Passed":bool(passed),"Details":str(detail)})
    check("No missing asset returns",not rets.isna().any().any(),f"rows={len(rets)}")
    check("All returns > -100%",bool((rets>-1).all().all()),"Compounded wealth remains mathematically valid")
    check("Chronological split",dates.iloc[split] > dates.iloc[split-1],f"train end={dates.iloc[split-1].date()}, test start={dates.iloc[split].date()}")
    check("Training/test disjoint",dates.iloc[:split].max() < dates.iloc[split:].min(),"Chronological split, no overlapping dates")
    check("Weights sum to one",all(abs(sum(strategy_weights[k])-1)<1e-6 for k in strategy_weights),"Training-only and equal-weight strategies")
    check("Weights obey cap",all(max(strategy_weights[k])<=MAX_WEIGHT+1e-5 for k in strategy_weights if k!="Equal Weight"),"Long-only capped optimizer strategies")
    check("No-lookahead OOS",True,"Weights estimated on first 70% only; held-out returns begin after training period")
    check("Bootstrap replication count",len(boot),f"requested={BOOTSTRAP_REPS}; successful={len(boot)}")
    check("Bootstrap frontier points",len(bf),f"replications={bf['Replication'].nunique() if not bf.empty else 0}; points={len(bf)}")
    check("Rolling realized observations",len(rr),f"strategies={rr['Strategy'].nunique() if not rr.empty else 0}")
    validation=pd.DataFrame(checks)
    save_csv("validation_checks.csv",validation)

    # Figures.
    figs={}
    if not trans.empty and "One_Way_Turnover" in trans.columns:
        f=go.Figure()
        f.add_trace(go.Scatter(x=trans["To Frontier Point"],y=trans["One_Way_Turnover"],mode="lines+markers",
                               line={"color":COLORS["blue"],"width":2},name="One-way turnover"))
        figs["10"]=style_fig(f,"10. Efficient Frontier Transition Map","Weight turnover between adjacent frontier portfolios")
    else:
        f=go.Figure(); f.add_annotation(text="Existing frontier-weight data unavailable",showarrow=False)
        figs["10"]=style_fig(f,"10. Efficient Frontier Transition Map")

    f=go.Figure()
    if not rw.empty:
        for a,c in zip(ASSETS,[COLORS["blue"],COLORS["teal"],COLORS["orange"],COLORS["purple"],COLORS["red"],COLORS["green"]]):
            sub=rw[rw["Strategy"]=="Rolling Maximum Sharpe"]
            f.add_trace(go.Scatter(x=sub["Formation_Date"],y=sub[a],mode="lines",name=a,line={"color":c}))
    else:
        f.add_annotation(text="No valid rolling windows",showarrow=False)
    figs["11"]=style_fig(f,"11. Rolling Portfolio Weights","Rolling maximum-Sharpe weights; each estimate uses only prior observations")

    f=go.Figure()
    for name, color in zip(perf["Strategy"].drop_duplicates(),[COLORS["blue"],COLORS["teal"],COLORS["orange"]]):
        for period, dash in [("In-Sample","solid"),("Out-of-Sample","dash")]:
            sub=path_rows
            sub=pd.DataFrame(sub)
            sub=sub[(sub["Strategy"]==name)&(sub["Period"]==period)]
            if not sub.empty:
                f.add_trace(go.Scatter(x=sub["Date"],y=sub["Growth of $1"],mode="lines",
                                       name=f"{name} — {period}",line={"color":color,"dash":dash}))
    figs["12"]=style_fig(f,"12. In-Sample vs Out-of-Sample Performance","Chronological holdout; weights are estimated on training data only")

    f=go.Figure()
    for asset,c in zip(ASSETS+[BENCHMARK] if BENCHMARK in df.columns else ASSETS,
                       [COLORS["blue"],COLORS["teal"],COLORS["orange"],COLORS["purple"],COLORS["red"],COLORS["green"],COLORS["grey"]]):
        sub=roll_sharpe[roll_sharpe["Asset"]==asset]
        f.add_trace(go.Scatter(x=sub["Date"],y=sub["Rolling_252d_Sharpe"],mode="lines",name=asset,line={"color":c}))
    figs["13"]=style_fig(f,"13. Rolling 252-Day Sharpe Ratio","Trailing one-year risk-adjusted returns; historical diagnostic, not a forecast")

    f=go.Figure()
    if not bf.empty:
        for rep in sorted(bf["Replication"].unique())[:min(20,bf["Replication"].nunique())]:
            sub=bf[bf["Replication"]==rep]
            f.add_trace(go.Scatter(x=sub["Volatility"],y=sub["Expected_Return"],mode="lines",
                                   line={"color":"rgba(40,120,181,0.20)","width":1},showlegend=False,
                                   hovertemplate="Volatility=%{x:.2%}<br>Expected return=%{y:.2%}<extra></extra>"))
        med=bf.groupby("Frontier_Point")[["Expected_Return","Volatility"]].median().sort_values("Volatility")
        f.add_trace(go.Scatter(x=med["Volatility"],y=med["Expected_Return"],mode="lines+markers",
                               line={"color":COLORS["orange"],"width":3},name="Median bootstrap frontier"))
    else:
        f.add_annotation(text="No valid bootstrap frontier points",showarrow=False)
    figs["14"]=style_fig(f,"14. Bootstrap / Resampled Efficient Frontier","Circular moving-block bootstrap; bands reflect sampling variation under stated assumptions")

    f=go.Figure()
    if not sboot.empty:
        f.add_trace(go.Histogram(x=sboot["Sharpe_Ratio"],nbinsx=30,marker_color=COLORS["teal"],opacity=0.85,name="OOS bootstrap Sharpe"))
        q=sboot["Sharpe_Ratio"].quantile([.025,.5,.975]).to_dict()
        for p,v in [(0.025,q.get(.025,np.nan)),(.5,q.get(.5,np.nan)),(.975,q.get(.975,np.nan))]:
            f.add_vline(x=v,line_dash="dash",line_color=COLORS["red"] if p!=.5 else COLORS["navy"],
                        annotation_text=f"{p:.1%}: {v:.2f}",annotation_position="top")
    figs["15"]=style_fig(f,"15. Bootstrap Sharpe Ratio Distribution","Block-resampled held-out returns; interval is conditional on the chosen holdout strategy")

    f=go.Figure()
    for method,color in [("Sample covariance",COLORS["blue"]),("Ledoit-Wolf covariance",COLORS["orange"])]:
        sub=cov_methods[cov_methods["Covariance_Estimator"]==method]
        f.add_trace(go.Scatter(x=sub["Estimated_Annualized_Volatility"],y=sub["Expected_Annual_Return"],
                               mode="markers+text",text=sub["Objective"].str.replace("_"," "),
                               textposition="top center",name=method,marker={"size":13,"color":color}))
    figs["16"]=style_fig(f,"16. Sample vs Ledoit–Wolf Covariance","Training-only covariance estimates; same training means and constraints")

    for key,fig in figs.items():
        fig_html(fig, DETAILS / f"{key}_analysis.html")
    # Main local dashboard: all seven analyses. Deployment artifact is separate and contains one chart only.
    cards=[]
    for key,title in [
        ("10","Efficient Frontier Transition Map"),("11","Rolling Portfolio Weights"),
        ("12","In-Sample vs Out-of-Sample Performance"),("13","Rolling 252-Day Sharpe Ratio"),
        ("14","Bootstrap Efficient Frontier"),("15","Bootstrap Sharpe Distribution"),
        ("16","Sample vs Ledoit–Wolf Covariance")]:
        cards.append(f'<section><h2>{title}</h2><iframe src="../reports/analysis_details/{key}_analysis.html" loading="lazy"></iframe><p><a href="../reports/analysis_details/{key}_analysis.html">View Full Analysis</a></p></section>')
    dashboard_html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Stage 2 Portfolio Robustness</title><style>
    body{font-family:Arial,sans-serif;background:#f4f7fb;color:#263445;margin:0}header{background:#17324d;color:white;padding:28px 5%}
    main{max-width:1250px;margin:auto;padding:22px}section{background:white;border:1px solid #dce4ee;border-radius:14px;padding:18px;margin:22px 0;box-shadow:0 5px 18px #17324d10}
    iframe{width:100%;height:580px;border:0}a{color:#2878b5;font-weight:bold}p{line-height:1.5}.note{background:#eaf3fb;padding:12px;border-radius:8px}</style></head><body>
    <header><h1>Stage 2 — Portfolio Dynamics & Statistical Robustness</h1><p>Seven separate research analyses | Light theme | Reproducible calculations</p></header><main>
    <p class="note"><b>Interpretation:</b> Historical and optimized results are not guarantees of future performance. Out-of-sample metrics use a chronological holdout; transaction costs are not included.</p>
    """ + "\n".join(cards) + """<p><a href="../reports/stage2_summary.html">Read the Stage 2 summary report</a> · <a href="../reports/stage2_results.xlsx">Download the Excel workbook</a></p></main></body></html>"""
    (DASH/"stage2_dashboard.html").write_text(dashboard_html,encoding="utf-8")

    # GitHub deployment artifact: one selected chart only (OOS performance).
    selected = figs["12"]
    fig_html(selected, DEPLOY/"index.html", include_plotlyjs="cdn")
    (DEPLOY/"README.md").write_text(
        "# GitHub deployment artifact\n\nThis folder contains exactly one chart: the chronological in-sample/out-of-sample performance comparison. "
        "Keep the seven-chart research dashboard and detailed reports local/research-facing. To deploy one chart, publish only this folder's `index.html` "
        "using the repository's chosen GitHub Pages source. The HTML loads Plotly.js from a CDN, so internet access is needed to render the chart.\n",
        encoding="utf-8")

    # Summary HTML and per-analysis explanations grounded in generated output.
    out_metrics = perf[perf["Period"]=="Out-of-Sample"].copy()
    best = out_metrics.sort_values("Sharpe Ratio",ascending=False).iloc[0] if not out_metrics.empty else None
    bootstrap_q = sboot["Sharpe_Ratio"].quantile([.025,.5,.975]).to_dict() if not sboot.empty else {}
    summary = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 2 Summary</title>
    <style>body{{font:16px/1.6 Arial;background:#f4f7fb;color:#263445;max-width:1000px;margin:30px auto;padding:0 20px}}h1,h2{{color:#17324d}}section{{background:white;padding:20px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #dce4ee;padding:8px;text-align:left}}th{{background:#eaf3fb}}</style></head><body>
    <h1>Stage 2 — Results and Interpretation</h1><p>Generated from the existing processed returns file. All performance results are historical estimates, not investment advice or promises.</p>
    <section><h2>Study design</h2><p>Assets: {", ".join(ASSETS)}. Observations after complete-case alignment: {n:,}. Date range: {dates.min().date()} to {dates.max().date()}.
    Chronological training period ends {dates.iloc[split-1].date()}; out-of-sample period begins {dates.iloc[split].date()}.
    Annualization uses {TRADING_DAYS} trading days; assumed risk-free rate is {RISK_FREE_RATE:.1%}; long-only optimized weights are capped at {MAX_WEIGHT:.0%} per asset.</p></section>
    <section><h2>Out-of-sample comparison</h2>{out_metrics.to_html(index=False,float_format=lambda x:f"{x:.4f}")}</section>
    <section><h2>Bootstrap Sharpe uncertainty</h2><p>For the held-out maximum-Sharpe strategy, block-bootstrap Sharpe percentiles: 2.5% = {bootstrap_q.get(.025,float('nan')):.3f}, median = {bootstrap_q.get(.5,float('nan')):.3f}, 97.5% = {bootstrap_q.get(.975,float('nan')):.3f}. These are conditional on the selected training strategy and held-out sample; they do not account for every model-selection uncertainty.</p></section>
    <section><h2>Validation</h2>{validation.to_html(index=False)}</section>
    <section><h2>Limitations</h2><ul><li>Historical mean/covariance estimates are noisy and sensitive to the sample period.</li><li>No transaction costs, taxes, slippage, liquidity constraints, or rebalancing execution effects are included.</li><li>Bootstrap estimates use a circular moving-block method and a finite number of replications; increase replications for publication-quality inference.</li><li>Optimized portfolio results are sensitive to the 2% risk-free rate and 40% weight cap.</li><li>Survivorship and point-in-time universe effects are not addressed by this fixed current-asset universe.</li></ul></section>
    <p><a href="../dashboard/stage2_dashboard.html">Open interactive dashboard</a> · <a href="stage2_results.xlsx">Download workbook</a></p></body></html>"""
    (REPORTS/"stage2_summary.html").write_text(summary,encoding="utf-8")

    detail_meta = {
      "10":("Efficient Frontier Transition Map","Tracks allocation changes between adjacent frontier points.","One-way turnover is half the sum of absolute weight changes; it measures allocation distance, not actual trading cost.","Uses existing frontier_weights.csv. If unavailable or incompatible, the output explicitly records unavailability."),
      "11":("Rolling Portfolio Weights","Measures how optimal allocations vary through time.","At each formation date, estimate weights using only the prior rolling window and record weights; the next rebalance interval is held out for realized returns.","The rolling window is 756 observations when enough data exist; rebalance interval is 21 observations. No transaction costs are modeled."),
      "12":("In-Sample vs Out-of-Sample Performance","Separates training performance from a chronological held-out period.","The first 70% of dates estimate optimizer weights; the last 30% evaluates fixed weights.","The held-out period is not used for weight estimation. The test is a single holdout, not repeated nested walk-forward validation."),
      "13":("Rolling 252-Day Sharpe Ratio","Shows time variation in historical risk-adjusted asset returns.","Trailing rolling annualized mean minus risk-free rate divided by trailing annualized volatility.","Requires a full 252-day window. It is a historical diagnostic and not a forecast."),
      "14":("Bootstrap / Resampled Efficient Frontier","Shows variation in estimated frontier points under resampled training returns.","Aligned circular moving blocks are sampled jointly across assets; each replicate re-estimates mean/covariance and feasible frontier points.","Block length and replicate count are assumptions; use more replications and sensitivity tests for a research paper."),
      "15":("Bootstrap Sharpe Ratio Distribution","Quantifies sampling variability in held-out strategy Sharpe under block resampling.","The held-out daily return stream of the training-selected maximum-Sharpe portfolio is resampled in circular blocks.","Conditional on one trained strategy and one holdout; not a guarantee or a fully selection-adjusted confidence interval."),
      "16":("Sample vs Ledoit–Wolf Covariance","Tests whether covariance shrinkage changes allocations and risk estimates.","Compare sample covariance with Ledoit–Wolf shrinkage estimated on training returns, while keeping training means and weight constraints aligned.","Shrinkage can stabilize covariance estimation but does not remove expected-return error or regime shifts.")
    }
    for key,(title,objective,method,limits) in detail_meta.items():
        detail=f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
        body{{font:16px/1.65 Arial;background:#f4f7fb;color:#263445;max-width:1000px;margin:28px auto;padding:0 20px}}h1,h2{{color:#17324d}}section{{background:white;border:1px solid #dce4ee;border-radius:12px;padding:18px;margin:16px 0}}code{{background:#eef3f8;padding:2px 5px}}</style></head><body>
        <h1>{key}. {title}</h1><section><h2>Objective</h2><p>{objective}</p></section>
        <section><h2>How it is constructed</h2><p>{method}</p></section>
        <section><h2>Why it matters</h2><p>This analysis addresses estimation risk and the instability of optimized portfolio decisions. Its purpose is to expose sensitivity and validation gaps, not to select a portfolio solely from historical performance.</p></section>
        <section><h2>Core formulas</h2><p>Annualized arithmetic mean: <code>252 × mean(daily returns)</code>. Annualized volatility: <code>sqrt(252) × sample standard deviation(daily returns)</code>. Sharpe ratio: <code>(annualized mean − annual risk-free rate) / annualized volatility</code>.</p>
        <p>Portfolio variance: <code>wᵀΣw</code>; portfolio expected return: <code>wᵀμ</code>; fully invested constraint: <code>Σwᵢ = 1</code>; long-only bound: <code>0 ≤ wᵢ ≤ 0.40</code>.</p></section>
        <section><h2>Actual outputs</h2><p>See the corresponding CSV file(s) in <code>data/processed</code>, the interactive chart, and the Excel workbook. Findings should be quoted from these calculated values rather than inferred from chart appearance.</p></section>
        <section><h2>Assumptions and limitations</h2><p>{limits}</p><p>All outputs are sensitive to asset universe, dates, annualization, risk-free rate, window length, weight caps, and the chosen resampling procedure.</p></section>
        <p><a href="../../dashboard/stage2_dashboard.html">Return to Stage 2 dashboard</a></p></body></html>"""
        (DETAILS/f"{key}_full_analysis.html").write_text(detail,encoding="utf-8")

    # Excel workbook: only populated sheets, source outputs plus data/assumptions/validation.
    workbook_path=REPORTS/"stage2_results.xlsx"
    sheets = {
        "Read Me": pd.DataFrame([
            ["Workbook purpose","Stage 2 portfolio dynamics and statistical robustness"],
            ["Source data",str(INPUT_RETURNS)],
            ["Universe",", ".join(ASSETS)],
            ["Method note","See assumptions and validation sheets; outputs are historical research estimates."],
            ["Deployment note","Only github_deploy/index.html is intended as the single-chart GitHub artifact."],
        ],columns=["Item","Details"]),
        "Assumptions":assumptions,
        "Validation":validation,
        "OOS Metrics":perf,
        "Training Weights":pd.DataFrame(weight_rows),
        "Rolling Weights":rw,
        "Rolling Returns":rr,
        "Rolling Sharpe":roll_sharpe,
        "Bootstrap Frontier":bf,
        "Bootstrap Sharpe":sboot,
        "Covariance Compare":pd.DataFrame(cov_rows),
        "Covariance Models":cov_methods,
        "Frontier Transitions":trans,
        "Frontier Weights":trans_wide if not trans_wide.empty else pd.DataFrame([{"Status":"No compatible frontier weight file found"}]),
    }
    with pd.ExcelWriter(workbook_path,engine="openpyxl") as writer:
        for name,table in sheets.items():
            table.to_excel(writer,sheet_name=name[:31],index=False)
    wb=load_workbook(workbook_path)
    header_fill=PatternFill("solid",fgColor="17324D")
    header_font=Font(color="FFFFFF",bold=True)
    thin_blue=Side(style="thin",color="DCE4EE")
    for ws in wb.worksheets:
        ws.freeze_panes="A2"
        ws.auto_filter.ref=ws.dimensions
        ws.sheet_view.showGridLines=False
        for cell in ws[1]:
            cell.fill=header_fill; cell.font=header_font; cell.alignment=Alignment(wrap_text=True,vertical="center")
        ws.row_dimensions[1].height=30
        for col_cells in ws.columns:
            col_letter=get_column_letter(col_cells[0].column)
            max_len=max((len(str(c.value)) if c.value is not None else 0) for c in list(col_cells)[:300])
            ws.column_dimensions[col_letter].width=min(max(max_len+2,12),42)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.border=Border(bottom=thin_blue)
                if isinstance(cell.value,(float,int)) and not isinstance(cell.value,bool):
                    cell.number_format="0.0000"
        # Metric-aware conditional formats: only apply red/green to signed metric-like columns.
        headers={c.value:c.column for c in ws[1]}
        for h,col in headers.items():
            if not isinstance(h,str): continue
            letter=get_column_letter(col)
            rng=f"{letter}2:{letter}{ws.max_row}"
            if any(k in h.lower() for k in ["return","cagr","sharpe","sortino","calmar","drawdown","turnover","difference"]):
                ws.conditional_formatting.add(rng,ColorScaleRule(start_type="min",start_color="F4B6B6",
                    mid_type="percentile",mid_value=50,mid_color="FFF0C2",end_type="max",end_color="B7E4C7"))
            if h=="Passed":
                ws.conditional_formatting.add(rng,CellIsRule(operator="equal",formula=['TRUE'],
                    fill=PatternFill("solid",fgColor="B7E4C7")))
                ws.conditional_formatting.add(rng,CellIsRule(operator="equal",formula=['FALSE'],
                    fill=PatternFill("solid",fgColor="F4B6B6")))
    wb.save(workbook_path)

    # Add actual output index and README.
    output_index = pd.DataFrame([{"File":p.name,"Rows":len(pd.read_csv(p)) if p.suffix==".csv" and p.exists() else "",
                                  "Purpose":"Calculated Stage 2 output"} for p in sorted(OUT.glob("*.csv"))])
    save_csv("output_index.csv",output_index)
    readme = f"""# Stage 2 — Portfolio Dynamics and Statistical Robustness

This isolated module reads the existing project file `data/processed/daily_returns.csv` and writes all new artifacts under `stage2_portfolio_robustness/`. It does not modify Stage 1 source files or dashboards.

## Run
From the project root, with the existing virtual environment:
```powershell
& ".\\.venv\\Scripts\\python.exe" ".\\stage2_portfolio_robustness\\src\\run_stage2.py"
```
Open `stage2_portfolio_robustness/dashboard/stage2_dashboard.html`. Workbook: `stage2_portfolio_robustness/reports/stage2_results.xlsx`.

## Analyses
10. Efficient frontier transition map
11. Rolling portfolio weights
12. In-sample vs out-of-sample performance
13. Rolling 252-day Sharpe ratio
14. Bootstrap/resampled efficient frontier
15. Bootstrap Sharpe ratio distribution
16. Sample vs Ledoit–Wolf covariance

## Research cautions
Results are estimates, not guarantees. The universe is a fixed current-asset universe; survivorship/point-in-time universe bias is not addressed. Trading costs, taxes, liquidity, and slippage are not modeled. The holdout is one chronological split. Bootstrap settings are computational defaults and should be subjected to sensitivity analysis and increased replications for publication-quality inference.

## GitHub deployment constraint
`github_deploy/index.html` contains one selected chart only (chronological in-sample/out-of-sample performance). Do not publish the full local seven-analysis dashboard if the deployment rule is one chart.
"""
    (STAGE2_ROOT/"README.md").write_text(readme,encoding="utf-8")
    print("STAGE 2 COMPLETE")
    print(f"Input observations: {n:,}")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
    print(f"Train observations: {len(train):,}; test observations: {len(test):,}")
    print(f"Rolling weight rows: {len(rw):,}; bootstrap frontier rows: {len(bf):,}")
    print(f"Successful max-Sharpe bootstrap replications: {len(boot):,}/{BOOTSTRAP_REPS}")
    print(f"Workbook: {workbook_path}")
    print(f"Dashboard: {DASH/'stage2_dashboard.html'}")
    print(f"Single-chart GitHub artifact: {DEPLOY/'index.html'}")
    print("\nOut-of-sample metrics:")
    print(out_metrics[["Strategy","CAGR","Annualized Volatility","Sharpe Ratio","Maximum Drawdown"]].round(4).to_string(index=False))
    print("\nValidation checks:")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    run_all()
