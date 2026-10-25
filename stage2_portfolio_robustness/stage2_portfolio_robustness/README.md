# Stage 2 — Portfolio Dynamics and Statistical Robustness

This isolated module reads the existing project file `data/processed/daily_returns.csv` and writes all new artifacts under `stage2_portfolio_robustness/`. It does not modify Stage 1 source files or dashboards.

## Run
From the project root, with the existing virtual environment:
```powershell
& ".\.venv\Scripts\python.exe" ".\stage2_portfolio_robustness\src\run_stage2.py"
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
