from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_returns.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.02
MAX_WEIGHT = 0.40
FRONTIER_POINTS = 60
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BND"]


def portfolio_return(weights, mean_returns):
    return float(np.asarray(weights) @ np.asarray(mean_returns))


def portfolio_volatility(weights, covariance):
    variance = float(np.asarray(weights) @ np.asarray(covariance) @ np.asarray(weights))
    return float(np.sqrt(max(variance, 0.0)))


def portfolio_sharpe(weights, mean_returns, covariance):
    volatility = portfolio_volatility(weights, covariance)
    if volatility <= 1e-12:
        return -np.inf
    return (portfolio_return(weights, mean_returns) - RISK_FREE_RATE) / volatility


def risk_parity_weights(covariance):
    """Approximate equal risk contributions with long-only and 40% per-asset bounds."""
    n = covariance.shape[0]
    initial = np.repeat(1.0 / n, n)

    def risk_contributions(w):
        variance = max(float(w @ covariance @ w), 1e-16)
        portfolio_vol = np.sqrt(variance)
        return w * (covariance @ w) / portfolio_vol

    def objective(w):
        rc = risk_contributions(w)
        target = rc.sum() / n
        return float(np.sum((rc - target) ** 2))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, MAX_WEIGHT)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"maxiter": 3000, "ftol": 1e-13},
    )
    if not result.success:
        raise RuntimeError(f"Risk-parity optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, MAX_WEIGHT)
    weights /= weights.sum()
    if np.any(weights < -1e-7) or np.any(weights > MAX_WEIGHT + 1e-5):
        raise RuntimeError("Risk-parity weights violate allocation bounds.")
    return weights


def calculate_metrics(daily_returns):
    r = pd.Series(np.asarray(daily_returns, dtype=float)).dropna()
    if r.empty:
        raise ValueError("Cannot calculate metrics from an empty return series.")

    wealth = np.cumprod(1.0 + r.to_numpy())
    years = len(r) / TRADING_DAYS
    cagr = float(wealth[-1] ** (1.0 / years) - 1.0) if wealth[-1] > 0 else np.nan
    arithmetic_annual_return = float(r.mean() * TRADING_DAYS)
    annual_volatility = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    downside = np.minimum(r.to_numpy(), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2) * TRADING_DAYS))
    max_drawdown = float((wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1.0).min())

    return {
        "CAGR": cagr,
        "Expected Annual Return": arithmetic_annual_return,
        "Historical Annualized Return": arithmetic_annual_return,
        "Historical Annualized Volatility": annual_volatility,
        "Sharpe Ratio": (
            (arithmetic_annual_return - RISK_FREE_RATE) / annual_volatility
            if annual_volatility > 1e-12 else np.nan
        ),
        "Sortino Ratio": (
            (arithmetic_annual_return - RISK_FREE_RATE) / downside_deviation
            if downside_deviation > 1e-12 else np.nan
        ),
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": cagr / abs(max_drawdown) if max_drawdown < -1e-12 else np.nan,
        "Total Return": float(wealth[-1] - 1.0),
        "Observations": int(len(r)),
    }


def check_weights(name, weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise RuntimeError(f"{name} weights contain non-finite values.")
    if abs(weights.sum() - 1.0) > 1e-5:
        raise RuntimeError(f"{name} weights sum to {weights.sum():.8f}, not 1.")
    if weights.min() < -1e-5 or weights.max() > MAX_WEIGHT + 1e-5:
        raise RuntimeError(f"{name} weights violate long-only or maximum-weight constraints.")
    return weights


def optimize_portfolios():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Returns file not found: {DATA_FILE}\n"
            "Run python .\\src\\download_data.py first."
        )

    returns = pd.read_csv(DATA_FILE, index_col="Date", parse_dates=True).sort_index()
    missing = sorted(set(ASSETS) - set(returns.columns))
    if missing:
        raise ValueError(f"Missing required assets: {missing}")

    returns = returns[ASSETS].replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < TRADING_DAYS:
        raise ValueError("At least one year of complete daily asset returns is required.")
    if (returns.std(ddof=1) <= 0).any():
        raise ValueError("At least one asset has zero return volatility.")
    if (returns <= -1.0).any().any():
        raise ValueError("Returns contain values <= -100%, which invalidate wealth compounding.")

    mean_returns = returns.mean().to_numpy() * TRADING_DAYS
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    covariance = (covariance + covariance.T) / 2.0
    if not np.isfinite(mean_returns).all() or not np.isfinite(covariance).all():
        raise ValueError("Expected returns or covariance contain non-finite values.")
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Estimated covariance matrix is not positive semidefinite.")

    n_assets = len(ASSETS)
    equal_weights = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, MAX_WEIGHT)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if n_assets * MAX_WEIGHT < 1.0:
        raise ValueError("The maximum-weight constraint makes a fully invested portfolio infeasible.")

    min_var_result = minimize(
        lambda w: float(w @ covariance @ w),
        equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-13},
    )
    if not min_var_result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {min_var_result.message}")
    min_var_weights = check_weights("Minimum Variance", min_var_result.x)

    sharpe_result = minimize(
        lambda w: -portfolio_sharpe(w, mean_returns, covariance),
        equal_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-13},
    )
    if not sharpe_result.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {sharpe_result.message}")
    max_sharpe_weights = check_weights("Maximum Sharpe", sharpe_result.x)

    max_return_result = linprog(
        c=-mean_returns,
        A_eq=np.ones((1, n_assets)),
        b_eq=[1.0],
        bounds=bounds,
        method="highs",
    )
    if not max_return_result.success:
        raise RuntimeError(f"Could not determine feasible return range: {max_return_result.message}")
    max_return_weights = check_weights("Maximum Return", max_return_result.x)

    gmv_return = portfolio_return(min_var_weights, mean_returns)
    maximum_return = portfolio_return(max_return_weights, mean_returns)
    if maximum_return < gmv_return - 1e-8:
        raise RuntimeError("Maximum feasible return is below the global minimum-variance portfolio return.")

    # Efficient upper branch: minimum variance at each feasible target return,
    # starting at the global minimum-variance portfolio's return.
    target_returns = np.linspace(gmv_return, maximum_return, FRONTIER_POINTS)
    frontier_rows = []
    frontier_weight_rows = []
    start_weights = min_var_weights.copy()

    for target_index, target in enumerate(target_returns):
        target_constraint = {
            "type": "eq",
            "fun": lambda w, target=target: portfolio_return(w, mean_returns) - target,
        }
        target_constraints = constraints + [target_constraint]
        starts = [start_weights, min_var_weights, max_return_weights]
        best = None
        for start in starts:
            result = minimize(
                lambda w: float(w @ covariance @ w),
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=target_constraints,
                options={"maxiter": 3000, "ftol": 1e-12},
            )
            if result.success:
                candidate = np.clip(result.x, 0.0, MAX_WEIGHT)
                candidate /= candidate.sum()
                actual_return = portfolio_return(candidate, mean_returns)
                if abs(actual_return - target) <= 2e-5:
                    if best is None or float(candidate @ covariance @ candidate) < float(best @ covariance @ best):
                        best = candidate
        if best is None:
            # Do not silently label a failed target as a valid frontier point.
            print(f"WARNING: target {target:.6f} did not converge; skipped.")
            continue

        start_weights = best
        volatility = portfolio_volatility(best, covariance)
        expected_return = portfolio_return(best, mean_returns)
        sharpe = (expected_return - RISK_FREE_RATE) / volatility if volatility > 1e-12 else np.nan
        frontier_rows.append({
            "Target Return": float(target),
            "Expected Return": expected_return,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
        })
        for asset, weight in zip(ASSETS, best):
            frontier_weight_rows.append({
                "Frontier Point": len(frontier_rows),
                "Target Return": float(target),
                "Asset": asset,
                "Weight": float(weight),
            })

    if len(frontier_rows) < max(10, int(FRONTIER_POINTS * 0.75)):
        raise RuntimeError(
            f"Only {len(frontier_rows)} of {FRONTIER_POINTS} frontier points converged. "
            "Review solver diagnostics before using the chart."
        )

    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(OUTPUT_DIR / "efficient_frontier.csv", index=False)
    pd.DataFrame(frontier_weight_rows).to_csv(OUTPUT_DIR / "frontier_weights.csv", index=False)

    strategies = {
        "Maximum Sharpe": max_sharpe_weights,
        "Minimum Variance": min_var_weights,
        "Equal Weight": equal_weights,
        "Risk Parity": risk_parity_weights(covariance),
    }

    metrics_rows = []
    weights_rows = []
    for strategy, weights in strategies.items():
        weights = check_weights(strategy, weights)
        daily_portfolio_returns = returns.to_numpy() @ weights
        stats = calculate_metrics(daily_portfolio_returns)
        stats["Strategy"] = strategy
        stats["Expected Annual Return"] = portfolio_return(weights, mean_returns)
        metrics_rows.append(stats)
        for asset, weight in zip(ASSETS, weights):
            weights_rows.append({"Strategy": strategy, "Asset": asset, "Weight": float(weight)})

    metrics = pd.DataFrame(metrics_rows)
    weights_df = pd.DataFrame(weights_rows)
    metrics.to_csv(OUTPUT_DIR / "portfolio_metrics.csv", index=False)
    weights_df.to_csv(OUTPUT_DIR / "portfolio_weights.csv", index=False)

    print("\nOPTIMIZATION COMPLETE")
    print(f"Return observations: {len(returns):,}")
    print(f"Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Assets: {', '.join(ASSETS)}")
    print(f"Risk-free rate assumption: {RISK_FREE_RATE:.2%}")
    print(f"Maximum weight per asset: {MAX_WEIGHT:.0%}")
    print(f"Frontier points calculated: {len(frontier_rows)} / {FRONTIER_POINTS}")
    print("\nPORTFOLIO METRICS (FULL-SAMPLE DESCRIPTIVE RESULTS)")
    cols = [
        "Strategy", "CAGR", "Expected Annual Return",
        "Historical Annualized Volatility", "Sharpe Ratio",
        "Sortino Ratio", "Maximum Drawdown", "Calmar Ratio", "Total Return",
    ]
    print(metrics[cols].round(4).to_string(index=False))
    print("\nMAXIMUM-SHARPE PORTFOLIO WEIGHTS")
    for asset, weight in zip(ASSETS, max_sharpe_weights):
        print(f"{asset:6s} {weight:8.2%}")
    print("\nSaved files:")
    for filename in [
        "efficient_frontier.csv", "frontier_weights.csv",
        "portfolio_metrics.csv", "portfolio_weights.csv",
    ]:
        print(OUTPUT_DIR / filename)
    print("\nInterpretation note: these optimized weights use the full sample. "
          "They are not out-of-sample or live-trading results.")


if __name__ == "__main__":
    optimize_portfolios()
