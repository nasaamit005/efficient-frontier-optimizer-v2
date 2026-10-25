# Efficient Frontier Optimizer

**Author:** Amit Singh  
**GitHub:** [@nasaamit005](https://github.com/nasaamit005)  
**Repository:** [efficient-frontier-optimizer-v2](https://github.com/nasaamit005/efficient-frontier-optimizer-v2)  
**Live project hub:** [Open the interactive project](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/)

A quantitative-finance research project for constrained mean–variance portfolio optimization, efficient-frontier analysis, portfolio diagnostics, and robustness evaluation. It separates **full-sample descriptive optimization** from **rolling and out-of-sample analyses** and provides interactive Plotly dashboards and analytical outputs.

> **Research note:** This is an educational and research implementation, not investment advice. Optimization results depend on the sample, estimated inputs, constraints, and assumptions. Passing software tests does not establish investment performance or publication readiness.

## Contents

- [Interactive dashboards](#interactive-dashboards)
- [Project questions](#project-questions)
- [Asset universe and assumptions](#asset-universe-and-assumptions)
- [Workflow](#workflow)
- [Methodology and equations](#methodology-and-equations)
- [Stage 1: optimization and diagnostics](#stage-1-optimization-and-diagnostics)
- [Stage 2: robustness analyses](#stage-2-robustness-analyses)
- [Outputs](#outputs)
- [Run locally](#run-locally)
- [Validation and testing](#validation-and-testing)
- [Interpretation and limitations](#interpretation-and-limitations)
- [Research extensions](#research-extensions)
- [References](#references)
- [Disclaimer](#disclaimer)

## Interactive dashboards

The published GitHub Pages links below are live.

| Resource | What it contains |
|---|---|
| [Project homepage](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/) | Main project hub |
| [Stage 1 — Efficient Frontier](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage1/efficient_frontier_dashboard.html) | Risk–return trade-offs and portfolio optimization |
| [Stage 1 — Market & Asset Diagnostics](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage1/market_asset_diagnostics_dashboard.html) | Asset/market diagnostics and supporting metrics |
| [Stage 2 — Portfolio Robustness Dashboard](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage2/dashboard/stage2_dashboard.html) | Rolling, out-of-sample, bootstrap, and covariance analyses |
| [Stage 2 — Results Summary](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage2/reports/stage2_summary.html) | Consolidated Stage 2 results |
| [Stage 2 — Results workbook](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage2/reports/stage2_results.xlsx) | Spreadsheet outputs for further inspection |

The dashboards are interactive HTML reports. Plotly charts that load the Plotly JavaScript library from a CDN require an internet connection.

## Project questions

1. How does estimated expected return trade off against estimated portfolio volatility under explicit allocation constraints?
2. How do maximum-Sharpe, minimum-variance, equal-weight, and risk-parity allocations differ?
3. How do portfolio results change when weights are re-estimated over time?
4. How do in-sample results compare with the project's out-of-sample evaluation?
5. How sensitive are frontier and allocation estimates to resampling and covariance-estimation choices?
6. What can rolling Sharpe ratios, drawdowns, and validation checks reveal that a single full-period performance number cannot?

## Asset universe and assumptions

The optimizer uses six assets:

| Ticker | Description |
|---|---|
| `AAPL` | Apple equity |
| `MSFT` | Microsoft equity |
| `GOOGL` | Alphabet equity |
| `AMZN` | Amazon equity |
| `META` | Meta Platforms equity |
| `BND` | Broad U.S. investment-grade bond ETF |

The benchmark `^GSPC` (S&P 500 index) is retained when present in the input data for analysis; it is not one of the six assets used to form the Stage 1 constrained portfolios.

| Setting | Implementation |
|---|---:|
| Annualization convention | 252 trading days |
| Risk-free rate | 2.0% per year |
| Long-only constraint | Each weight is at least 0 |
| Maximum asset weight | 40% |
| Fully invested constraint | Weights sum to 1 |
| Stage 1 frontier targets | 60 |
| Stage 2 rolling window | 756 observations |
| Stage 2 rebalance step | 21 observations |
| Stage 2 bootstrap replications | 60 |
| Stage 2 bootstrap block length | 21 observations |
| Random seed | 42 |

These are code-level settings, not universal financial truths. The 2% risk-free rate is an assumption, and a 21-observation block is approximately one trading month under the 252-day convention.

## Workflow

```text
Daily return data
       |
       v
Data checks and asset selection
       |
       +------------------------------+
       |                              |
       v                              v
Stage 1: Full-sample              Stage 2: Robustness
optimization and diagnostics     and time-aware evaluation
       |                              |
       v                              v
Frontier + strategy weights      Rolling / OOS / bootstrap /
and descriptive metrics          covariance comparisons
       |                              |
       +---------------+--------------+
                       v
              CSV / Excel / HTML
                       |
                       v
             Interactive GitHub Pages
```

## Methodology and equations

Let there be N assets, with portfolio weights $\\mathbf{w}$, estimated expected returns $\\boldsymbol{\\mu}$, and estimated return covariance matrix $\\boldsymbol{\\Sigma}$.

### Portfolio expected return

$$
\\mu_p = \\mathbf{w}^{\\mathsf T}\\boldsymbol{\\mu}
$$

Here, $w_i$ is the fraction invested in asset i, and $\\mu_i$ is its estimated annualized arithmetic mean return.

### Portfolio variance and volatility

$$
\\sigma_p^2 = \\mathbf{w}^{\\mathsf T}\\boldsymbol{\\Sigma}\\mathbf{w},
\\qquad
\\sigma_p = \\sqrt{\\mathbf{w}^{\\mathsf T}\\boldsymbol{\\Sigma}\\mathbf{w}}
$$

Covariance terms matter: portfolio risk is not simply the weighted average of individual asset volatilities.

### Fully invested, long-only, capped allocation

$$
\\sum_{i=1}^{N} w_i = 1,
\\qquad
0 \\leq w_i \\leq 0.40
$$

These constraints prohibit short selling and cap each individual asset at 40%. The cap can materially change the attainable frontier and optimal allocation.

### Maximum-Sharpe portfolio

$$
\\max_{\\mathbf{w}}
\\quad
\\frac{\\mathbf{w}^{\\mathsf T}\\boldsymbol{\\mu}-r_f}
{\\sqrt{\\mathbf{w}^{\\mathsf T}\\boldsymbol{\\Sigma}\\mathbf{w}}}
$$

Here, $r_f$ is the assumed annual risk-free rate (2%). The optimizer uses constrained numerical optimization (SLSQP). The result can be sensitive to estimation error and numerical behavior.

### Minimum-variance portfolio

$$
\\min_{\\mathbf{w}}
\\quad \\mathbf{w}^{\\mathsf T}\\boldsymbol{\\Sigma}\\mathbf{w}
$$

subject to the fully invested and weight-bound constraints.

### Efficient frontier

For a target expected return $\\mu^*$, the implementation minimizes variance:

$$
\\min_{\\mathbf{w}}
\\quad \\mathbf{w}^{\\mathsf T}\\boldsymbol{\\Sigma}\\mathbf{w}
\\quad \\text{subject to}\\quad
\\mathbf{w}^{\\mathsf T}\\boldsymbol{\\mu}=\\mu^*
$$

Stage 1 creates 60 target returns and records a frontier point only when the numerical solver converges and the achieved return is sufficiently close to its target. It raises an error if fewer than 75% of requested points (or fewer than 10) converge.

### Historical performance metrics

For daily portfolio returns $r_t$, compounded wealth is:

$$
W_T = \\prod_{t=1}^{T}(1+r_t)
$$

The annualized compound growth rate is calculated using the number of observations and the 252-trading-day convention:

$$
\\mathrm{CAGR}=W_T^{252/T}-1
$$

Annualized volatility uses the sample standard deviation of daily returns:

$$
\\sigma_{\\mathrm{ann}}=s(r_t)\\sqrt{252}
$$

The implementation's Sharpe ratio uses annualized arithmetic mean return less the risk-free-rate assumption, divided by annualized volatility:

$$
\\mathrm{Sharpe}=
\\frac{252\\bar r-r_f}{s(r_t)\\sqrt{252}}
$$

The Sortino calculation uses downside deviation based on negative daily returns. Maximum drawdown is the minimum observed decline from the prior running wealth peak. Calmar ratio is CAGR divided by the absolute maximum drawdown when drawdown is non-zero.

**Metric convention matters:** CAGR and annualized arithmetic mean return are different statistics and should not be treated as interchangeable.

## Stage 1: optimization and diagnostics

Stage 1 reads the processed daily-return dataset and computes annualized arithmetic means and the sample covariance matrix. It symmetrizes the covariance matrix and checks positive-semidefinite behavior before optimization.

Implemented portfolio comparisons include:

- **Maximum Sharpe:** seeks the highest estimated excess-return-to-volatility ratio.
- **Minimum variance:** minimizes estimated variance under the same weight constraints.
- **Equal weight:** a transparent baseline allocation.
- **Risk parity:** uses constrained numerical optimization to approximately equalize asset-level risk contributions.

The main outputs include the efficient-frontier table, frontier-point weights, portfolio metrics, and strategy weights. The dashboards add visual diagnostics for interpreting risk–return trade-offs and asset behavior.

## Stage 2: robustness analyses

Stage 2 expands the analysis beyond a single full-sample allocation. Its saved outputs and report pages cover:

| Analysis area | Saved outputs | Purpose |
|---|---|---|
| Frontier allocation transition | `10_frontier_transition_map.csv`, `10_frontier_weights_wide.csv` | Examine allocation variation across frontier targets |
| Rolling portfolio weights and returns | `11_rolling_portfolio_weights.csv`, `11_rolling_realized_returns.csv` | Study allocation changes and subsequent realized returns |
| In-sample vs out-of-sample evaluation | `12_insample_outofsample_metrics.csv`, `12_performance_paths.csv`, `12_training_weights.csv` | Compare training estimates with held-out evaluation |
| Rolling 252-day Sharpe | `13_rolling_252d_sharpe.csv` | Observe changes in rolling risk-adjusted performance |
| Bootstrap frontier | `14_bootstrap_frontier_points.csv` | Examine variation in frontier estimates under resampling |
| Bootstrap Sharpe diagnostics | `15_bootstrap_sharpe_distribution.csv`, `15_oos_sharpe_block_bootstrap.csv` | Examine resampled Sharpe distributions and OOS Sharpe uncertainty |
| Covariance comparison | `16_covariance_comparison.csv`, `16_covariance_optimizer_comparison.csv` | Compare covariance estimators and optimization choices |
| Assumptions and checks | `assumptions.csv`, `output_index.csv`, `validation_checks.csv` | Record run assumptions, outputs, and validation checks |

The Stage 2 engine includes a Ledoit–Wolf covariance-shrinkage estimator and a block-bootstrap index generator. Current settings use 60 bootstrap replications and blocks of 21 observations. Treat these as **exploratory diagnostics**, not definitive evidence of stable performance or precise confidence intervals. A research paper should increase replication counts and assess sensitivity to block length and sampling design.

Out-of-sample evaluation is more informative than optimizing and evaluating on the same observations, but it does not by itself establish deployability. Results remain sensitive to the split design, rebalancing assumptions, trading costs, and repeated model selection.

## Outputs

The primary Stage 1 processed outputs are:

- `data/processed/efficient_frontier.csv`
- `data/processed/frontier_weights.csv`
- `data/processed/portfolio_metrics.csv`
- `data/processed/portfolio_weights.csv`

Stage 2 outputs are stored under the Stage 2 project tree, including the CSVs listed above. Published HTML reports and the results workbook are available through the [Stage 2 dashboard](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage2/dashboard/stage2_dashboard.html) and [results summary](https://nasaamit005.github.io/efficient-frontier-optimizer-v2/stage2/reports/stage2_summary.html).

## Run locally

Commands below assume Windows PowerShell, the project root, and a working Python 3.12 environment. Use the existing environment if already configured.

```powershell
# From the project root
python --version

# Install dependencies if a requirements file exists
python -m pip install -r .\\requirements.txt

# Run Stage 1 optimization
python .\\src\\optimization.py
```

The Stage 1 optimizer expects `data\\processed\\daily_returns.csv` with a `Date` column and the six required asset columns. The exact data-download and preprocessing command depends on the scripts included in the repository; inspect the current source tree before assuming a particular entry point.

Stage 1 writes the four CSV files listed in [Outputs](#outputs). Stage 2's engine is part of the `stage2_recovery_original\\stage2_portfolio_robustness` tree. Before running it locally, inspect the available runner script and its documented input/output paths; published reports are already available through GitHub Pages.

## Validation and testing

Stage 2 tests were run from the project root:

```powershell
python -m pytest .\\stage2_recovery_original\\stage2_portfolio_robustness\\tests -q
```

**Verified result:** `4 passed in 24.12s`.

This confirms that the four tests in that test directory passed in that run. It does not mean every possible data condition, optimizer configuration, statistical assumption, or deployment scenario has been tested. For a research submission, document test coverage, edge cases, repeated-run reproducibility, and sensitivity checks separately.

## Interpretation and limitations

- **Estimation error:** expected returns are difficult to estimate, and small changes can produce large changes in maximum-Sharpe weights.
- **Full-sample bias:** Stage 1 strategy metrics are descriptive full-sample results; optimized weights are not live-trading results.
- **Out-of-sample is not automatically prospective:** a historical holdout can still be affected by repeated model selection.
- **Transaction costs and turnover:** results should not be treated as net-of-cost unless costs are explicitly deducted. Turnover and market impact need separate modelling.
- **Rebalancing:** practical outcomes depend on when weights are updated and how trades are executed.
- **Covariance assumptions:** historical covariance may be unstable; shrinkage is one estimator, not a guarantee of superior future performance.
- **Bootstrap limits:** 60 replications are too few for strong tail inference. Results may depend on block length and bootstrap design.
- **Model and data risk:** corporate actions, data revisions, survivorship, missing values, and instrument differences can affect results.
- **Benchmark interpretation:** the S&P 500 benchmark is for comparison when present; it is not part of the six-asset optimized universe.

## Research extensions

1. Increase bootstrap replications substantially (for example, 1,000–5,000 or more where computationally practical) and report Monte Carlo uncertainty.
2. Perform sensitivity analysis over risk-free rates, estimation windows, rebalancing intervals, weight caps, and asset universes.
3. Include explicit commissions, spreads, slippage, turnover, and market-impact assumptions.
4. Compare sample covariance with multiple shrinkage estimators and robust or factor-based covariance models.
5. Compare mean–variance portfolios with 1/N, minimum variance, risk parity, and other pre-specified baselines using the same out-of-sample protocol.
6. Report confidence intervals and use dependence-aware inference where appropriate; financial returns are not generally independent observations.
7. Control for repeated strategy selection and backtest overfitting; preserve a genuinely untouched final evaluation period.
8. Record software versions, data provenance, exact date ranges, and random seeds in a reproducibility manifest.
9. Add tests for weight constraints, covariance edge cases, missing data, solver failures, and consistency between exported tables and dashboard values.

## References

These references provide foundational theory and research context; they are not evidence that every method has been implemented in this repository.

1. Markowitz, H. (1952). “Portfolio Selection.” *The Journal of Finance*, 7(1), 77–91. https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
2. Sharpe, W. F. (1964). “Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk.” *The Journal of Finance*, 19(3), 425–442. https://doi.org/10.1111/j.1540-6261.1964.tb02865.x
3. Ledoit, O., & Wolf, M. (2004). “Honey, I Shrunk the Sample Covariance Matrix.” *The Journal of Portfolio Management*, 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110
4. Michaud, R. O. (1989). “The Markowitz Optimization Enigma: Is ‘Optimized’ Optimal?” *Financial Analysts Journal*, 45(1), 31–42. https://doi.org/10.2469/faj.v45.n1.31
5. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). “Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?” *The Review of Financial Studies*, 22(5), 1915–1953. https://doi.org/10.1093/rfs/hhm075
6. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
7. Politis, D. N., & Romano, J. P. (1994). “The Stationary Bootstrap.” *Journal of the American Statistical Association*, 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870
8. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). “The Probability of Backtest Overfitting.” *The Journal of Computational Finance*, 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322

## Disclaimer

For educational and research purposes only. This repository is not investment advice, does not guarantee future performance, and should not be used as the sole basis for investment decisions.
