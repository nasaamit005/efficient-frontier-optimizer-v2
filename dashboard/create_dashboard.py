
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "efficient_frontier_dashboard.html"

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BND"]
BENCHMARK = "^GSPC"
STRATEGIES = ["Maximum Sharpe", "Minimum Variance", "Equal Weight", "Risk Parity"]
RF = 0.02
CAP = 0.40
TRADING_DAYS = 252
ROLLING_WINDOW = 126
SEED = 42

ASSET_COLORS = {
    "AAPL": "#2563EB", "MSFT": "#0F766E", "GOOGL": "#7C3AED",
    "AMZN": "#D97706", "META": "#DB2777", "BND": "#64748B",
    "^GSPC": "#334155"
}
STRATEGY_COLORS = {
    "Maximum Sharpe": "#047857",
    "Minimum Variance": "#7C3AED",
    "Equal Weight": "#2563EB",
    "Risk Parity": "#D97706"
}
BG = "#F7F9FC"
GRID = "#E5EAF1"
TEXT = "#243447"


def load_returns():
    path = DATA / "daily_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/portfolio_metrics.py first.")

    df = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")),
        None
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.set_index(date_col)
    else:
        first = df.columns[0]
        parsed = pd.to_datetime(df[first], errors="coerce")
        if parsed.notna().mean() > 0.8:
            df[first] = parsed
            df = df.set_index(first)
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.loc[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    required = ASSETS + [BENCHMARK]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in daily_returns.csv: {missing}")

    return df[required].replace([np.inf, -np.inf], np.nan).dropna()


def load_frontier():
    path = DATA / "efficient_frontier.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/optimization.py first.")
    df = pd.read_csv(path)
    required = ["Expected Return", "Volatility"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Frontier file needs columns: {required}")
    return df.sort_values("Volatility")


def load_weights():
    path = DATA / "portfolio_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run src/optimization.py first.")

    df = pd.read_csv(path)
    required = {"Strategy", "Asset", "Weight"}
    if not required.issubset(df.columns):
        raise ValueError(
            "portfolio_weights.csv must contain Strategy, Asset and Weight."
        )

    result = {}
    for strategy in STRATEGIES:
        part = df[df["Strategy"].astype(str).str.strip() == strategy]
        if part.empty:
            raise ValueError(f"No saved weights found for strategy: {strategy}")
        weights = part.set_index("Asset")["Weight"].reindex(ASSETS).fillna(0.0)
        weights = weights.astype(float).to_numpy()
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError(f"Invalid weights for {strategy}")
        result[strategy] = weights / weights.sum()

    return result


def portfolio_point(weights, mean_returns, covariance):
    expected_return = float(weights @ mean_returns)
    variance = float(weights @ covariance @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    sharpe = (expected_return - RF) / volatility if volatility > 0 else np.nan
    return expected_return, volatility, sharpe


def max_drawdown(series):
    wealth = (1 + series).cumprod()
    wealth = pd.concat([pd.Series([1.0], index=[series.index[0] -
                            pd.Timedelta(days=1)]), wealth]).sort_index()
    drawdown = wealth / wealth.cummax() - 1
    return drawdown.iloc[1:]


def performance_stats(series):
    series = series.dropna()
    wealth = (1 + series).cumprod()
    years = len(series) / TRADING_DAYS
    total_return = float(wealth.iloc[-1] - 1) if len(wealth) else np.nan
    cagr = (
        float(wealth.iloc[-1] ** (1 / years) - 1)
        if len(wealth) and years > 0 and wealth.iloc[-1] > 0 else np.nan
    )
    annual_vol = float(series.std(ddof=1) * np.sqrt(TRADING_DAYS))
    annual_mean = float(series.mean() * TRADING_DAYS)
    sharpe = (annual_mean - RF) / annual_vol if annual_vol > 0 else np.nan
    downside = np.minimum(series.to_numpy(), 0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS))
    sortino = (annual_mean - RF) / downside_dev if downside_dev > 0 else np.nan
    dd = max_drawdown(series)
    mdd = float(dd.min()) if len(dd) else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {
        "CAGR": cagr, "Annualized Mean": annual_mean,
        "Annualized Volatility": annual_vol, "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino, "Maximum Drawdown": mdd,
        "Calmar Ratio": calmar, "Total Return": total_return
    }



def feasible_cloud(mean_returns, covariance, n=3500):
    """Sample long-only portfolios respecting the per-asset weight cap."""
    rng = np.random.default_rng(SEED)
    accepted = []
    accepted_count = 0
    max_batches = 100

    for _ in range(max_batches):
        if accepted_count >= n:
            break

        weights = rng.dirichlet(
            np.full(len(ASSETS), 3.0),
            size=8000
        )
        weights = weights[
            np.max(weights, axis=1) <= CAP + 1e-10
        ]

        if len(weights):
            accepted.append(weights)
            accepted_count += len(weights)

    if accepted_count < n:
        raise RuntimeError(
            "Could not generate enough feasible portfolios. "
            "Check the asset count and maximum-weight constraint."
        )

    weights = np.vstack(accepted)[:n]
    expected_returns = weights @ mean_returns

    variances = np.einsum(
        "ij,jk,ik->i", weights, covariance, weights
    )
    volatility = np.sqrt(np.maximum(variances, 0.0))

    sharpe = np.divide(
        expected_returns - RF,
        volatility,
        out=np.zeros_like(expected_returns),
        where=volatility > 0
    )

    return pd.DataFrame({
        "Return": expected_returns,
        "Volatility": volatility,
        "Sharpe": sharpe
    })




def apply_layout(fig, title, subtitle=None, height=610):
    fig.update_layout(
        title=dict(
            text=title,
            x=0.025,
            y=0.99,
            xanchor="left",
            yanchor="top",
            font=dict(size=22, color=TEXT)
        ),
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial, sans-serif",
            color=TEXT,
            size=12
        ),
        height=height,
        margin=dict(
            l=80,
            r=65,
            t=190 if subtitle else 100,
            b=85
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.055,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E5EAF1",
            borderwidth=1,
            font=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        ),
        hovermode="closest"
    )

    if subtitle:
        fig.add_annotation(
            text=subtitle,
            xref="paper",
            yref="paper",
            x=0,
            y=1.19,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            align="left",
            font=dict(size=12, color="#64748B")
        )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        automargin=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        automargin=True
    )



def make_frontier_chart(returns, frontier, weights_map, mean_returns, covariance):
    fig = go.Figure()
    cloud = feasible_cloud(mean_returns, covariance)
    fig.add_trace(go.Scatter(
        x=cloud["Volatility"], y=cloud["Return"],
        mode="markers", name="Simulated feasible portfolios",
        marker=dict(
            size=5, color=cloud["Sharpe"], colorscale="Viridis",
            showscale=True, colorbar=dict(
                title="Sharpe<br>ratio", thickness=13, len=0.62,
                x=1.02
            ), opacity=0.52
        ),
        customdata=np.column_stack([cloud["Sharpe"]]),
        hovertemplate=(
            "<b>Feasible portfolio</b><br>"
            "Expected return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<br>"
            "Sharpe ratio: %{customdata[0]:.3f}<extra></extra>"
        )
    ))

    f = frontier.copy()
    if "Sharpe Ratio" not in f.columns:
        f["Sharpe Ratio"] = (
            f["Expected Return"] - RF
        ) / f["Volatility"].replace(0, np.nan)

    fig.add_trace(go.Scatter(
        x=f["Volatility"], y=f["Expected Return"],
        mode="lines", name="Efficient frontier",
        line=dict(color="#0F766E", width=4),
        customdata=f[["Sharpe Ratio"]].to_numpy(),
        hovertemplate=(
            "<b>Efficient frontier</b><br>"
            "Expected return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<br>"
            "Sharpe ratio: %{customdata[0]:.3f}<extra></extra>"
        )
    ))

    # Plot the six individual assets using estimated moments.
    asset_mu = returns[ASSETS].mean() * TRADING_DAYS
    asset_cov = returns[ASSETS].cov() * TRADING_DAYS
    asset_vol = np.sqrt(np.diag(asset_cov))
    asset_sharpe = (asset_mu - RF) / asset_vol

    fig.add_trace(go.Scatter(
        x=asset_vol, y=asset_mu,
        mode="markers+text", name="Individual assets",
        text=ASSETS, textposition="top center",
        marker=dict(
            size=10, color=[ASSET_COLORS[a] for a in ASSETS],
            line=dict(color="white", width=1.2), symbol="circle"
        ),
        customdata=np.column_stack([asset_sharpe]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Expected return: %{y:.2%}<br>"
            "Volatility: %{x:.2%}<br>"
            "Sharpe ratio: %{customdata[0]:.3f}<extra></extra>"
        )
    ))

    # Add Equal Weight and Risk Parity; GMV and Max-Sharpe get larger
    # dedicated markers and annotations below.
    for strategy, symbol in [
        ("Equal Weight", "square"),
        ("Risk Parity", "triangle-up")
    ]:
        w = weights_map[strategy]
        ret, vol, sharpe = portfolio_point(w, mean_returns, covariance)
        fig.add_trace(go.Scatter(
            x=[vol], y=[ret], mode="markers+text", name=strategy,
            text=[strategy], textposition="bottom right",
            marker=dict(
                size=13, symbol=symbol,
                color=STRATEGY_COLORS[strategy],
                line=dict(color="white", width=1.5)
            ),
            customdata=[[sharpe]],
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "Expected return: %{y:.2%}<br>"
                "Volatility: %{x:.2%}<br>"
                "Sharpe ratio: %{customdata[0]:.3f}<extra></extra>"
            )
        ))

    # Global Minimum-Variance portfolio: Minimum Variance under constraints.
    gmv_w = weights_map["Minimum Variance"]
    gmv_ret, gmv_vol, gmv_sharpe = portfolio_point(
        gmv_w, mean_returns, covariance
    )
    fig.add_trace(go.Scatter(
        x=[gmv_vol], y=[gmv_ret],
        mode="markers+text", name="GMV (Minimum Variance)",
        text=["GMV"], textposition="bottom left",
        marker=dict(
            size=20, symbol="circle",
            color="#7C3AED", line=dict(color="white", width=2.5)
        ),
        customdata=[[gmv_ret, gmv_vol, gmv_sharpe]],
        hovertemplate=(
            "<b>Global Minimum-Variance (GMV)</b><br>"
            "Expected return: %{customdata[0]:.2%}<br>"
            "Volatility: %{customdata[1]:.2%}<br>"
            "Sharpe ratio: %{customdata[2]:.3f}<extra></extra>"
        )
    ))

    # Maximum-Sharpe portfolio under the project's long-only/weight-cap constraints.
    max_w = weights_map["Maximum Sharpe"]
    max_ret, max_vol, max_sharpe = portfolio_point(
        max_w, mean_returns, covariance
    )
    fig.add_trace(go.Scatter(
        x=[max_vol], y=[max_ret],
        mode="markers+text", name="Maximum-Sharpe",
        text=["MAX SHARPE"], textposition="top right",
        marker=dict(
            size=22, symbol="diamond",
            color="#047857", line=dict(color="white", width=2.5)
        ),
        customdata=[[max_ret, max_vol, max_sharpe]],
        hovertemplate=(
            "<b>Maximum-Sharpe Portfolio</b><br>"
            "Expected return: %{customdata[0]:.2%}<br>"
            "Volatility: %{customdata[1]:.2%}<br>"
            "Sharpe ratio: %{customdata[2]:.3f}<extra></extra>"
        )
    ))

    # Risk-free reference line through the constrained Maximum-Sharpe portfolio.
    x_end = max(float(f["Volatility"].max()), max_vol) * 1.12
    x_line = np.linspace(0, x_end, 80)
    y_line = RF + max_sharpe * x_line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        name="2% risk-free reference",
        line=dict(color="#64748B", width=1.8, dash="dash"),
        hovertemplate=(
            "Reference volatility: %{x:.2%}<br>"
            "Reference return: %{y:.2%}<extra></extra>"
        )
    ))
    fig.add_hline(
        y=RF, line_width=1, line_dash="dot", line_color="#94A3B8",
        annotation_text="Risk-free rate = 2.00%",
        annotation_position="bottom right"
    )

    apply_layout(
        fig,
        "Efficient Frontier & Optimal Portfolio Analysis",
        "Annualized estimates · Long-only portfolios · Maximum 40% per asset · Risk-free rate 2.00%",
        height=720
    )
    fig.update_xaxes(
        title="Annualized volatility (risk)",
        tickformat=".0%", rangemode="tozero"
    )
    fig.update_yaxes(
        title="Expected annual return",
        tickformat=".0%"
    )

    # Explain the efficient/inefficient distinction without fabricating a
    # lower-branch optimization that is not in the exported frontier file.
    fig.add_annotation(
        x=0.02,
        y=0.96,
        xref="paper",
        yref="paper",
        text=(
            "<b>Efficient branch:</b> highest estimated return for a "
            "given risk level.<br>"
            "<b>Inefficient region:</b> portfolios dominated by alternatives "
            "offering higher return at comparable risk."
        ),
        showarrow=False,
        align="left",
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor=GRID,
        borderwidth=1,
        borderpad=8,
        font=dict(size=10, color=TEXT)
    )

    return fig, {
        "GMV": (gmv_ret, gmv_vol, gmv_sharpe),
        "Maximum Sharpe": (max_ret, max_vol, max_sharpe)
    }


def make_allocation_chart(weights_map):
    fig = go.Figure()
    for strategy in STRATEGIES:
        w = weights_map[strategy]
        fig.add_trace(go.Bar(
            x=ASSETS, y=w, name=strategy,
            marker_color=STRATEGY_COLORS[strategy],
            hovertemplate=(
                f"<b>{strategy}</b><br>"
                "Asset: %{x}<br>Weight: %{y:.2%}<extra></extra>"
            )
        ))
    fig.update_layout(barmode="group")
    apply_layout(
        fig, "Strategic Asset Allocation",
        "Compare the asset weights selected by all four portfolio strategies."
    )
    fig.update_xaxes(title="Asset")
    fig.update_yaxes(title="Portfolio weight", tickformat=".0%", rangemode="tozero")
    return fig


def make_growth_chart(returns, strategy_returns):
    fig = go.Figure()
    for strategy in STRATEGIES:
        growth = (1 + strategy_returns[strategy]).cumprod()
        fig.add_trace(go.Scatter(
            x=growth.index, y=growth, mode="lines", name=strategy,
            line=dict(color=STRATEGY_COLORS[strategy], width=2),
            hovertemplate=(
                f"<b>{strategy}</b><br>Date: %{{x|%Y-%m-%d}}<br>"
                "Growth of $1: $%{y:.3f}<extra></extra>"
            )
        ))
    benchmark = (1 + returns[BENCHMARK]).cumprod()
    fig.add_trace(go.Scatter(
        x=benchmark.index, y=benchmark, mode="lines",
        name="S&P 500 (^GSPC)",
        line=dict(color=ASSET_COLORS[BENCHMARK], width=2.5, dash="dash"),
        hovertemplate=(
            "<b>S&P 500</b><br>Date: %{x|%Y-%m-%d}<br>"
            "Growth of $1: $%{y:.3f}<extra></extra>"
        )
    ))
    apply_layout(
        fig, "Historical Portfolio Performance",
        "Growth of $1 invested at the start of the available sample; historical results are not forecasts."
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Portfolio value ($)", tickprefix="$", rangemode="tozero")
    return fig


def make_drawdown_chart(strategy_returns, returns):
    fig = go.Figure()
    for strategy in STRATEGIES:
        dd = max_drawdown(strategy_returns[strategy])
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, mode="lines", name=strategy,
            line=dict(color=STRATEGY_COLORS[strategy], width=1.8),
            hovertemplate=(
                f"<b>{strategy}</b><br>Date: %{{x|%Y-%m-%d}}<br>"
                "Drawdown: %{y:.2%}<extra></extra>"
            )
        ))
    dd_bench = max_drawdown(returns[BENCHMARK])
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench, mode="lines",
        name="S&P 500 (^GSPC)",
        line=dict(color=ASSET_COLORS[BENCHMARK], width=2, dash="dash"),
        hovertemplate=(
            "<b>S&P 500</b><br>Date: %{x|%Y-%m-%d}<br>"
            "Drawdown: %{y:.2%}<extra></extra>"
        )
    ))
    apply_layout(
        fig, "Drawdown & Downside Risk",
        "Drawdown measures the decline from a previous wealth peak; deeper troughs indicate larger losses from prior peaks."
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Drawdown from prior peak", tickformat=".0%", range=[None, 0])
    return fig


def make_rolling_vol_chart(strategy_returns, returns):
    fig = go.Figure()
    for strategy in STRATEGIES:
        rolling = (
            strategy_returns[strategy]
            .rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
        )
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling, mode="lines", name=strategy,
            line=dict(color=STRATEGY_COLORS[strategy], width=2),
            hovertemplate=(
                f"<b>{strategy}</b><br>Date: %{{x|%Y-%m-%d}}<br>"
                "Annualized volatility: %{y:.2%}<extra></extra>"
            )
        ))
    benchmark_rolling = (
        returns[BENCHMARK].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
    )
    fig.add_trace(go.Scatter(
        x=benchmark_rolling.index, y=benchmark_rolling,
        mode="lines", name="S&P 500 (^GSPC)",
        line=dict(color=ASSET_COLORS[BENCHMARK], width=2, dash="dash"),
        hovertemplate=(
            "<b>S&P 500</b><br>Date: %{x|%Y-%m-%d}<br>"
            "Annualized volatility: %{y:.2%}<extra></extra>"
        )
    ))
    apply_layout(
        fig, "Rolling Volatility (126 Trading Days)",
        "Annualized standard deviation of daily returns over a rolling 126-trading-day window."
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Annualized volatility", tickformat=".0%")
    return fig


def make_correlation_chart(returns):
    corr = returns[ASSETS].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.to_numpy(), x=ASSETS, y=ASSETS,
        zmin=-1, zmax=1, zmid=0,
        colorscale=[
            [0.0, "#B91C1C"], [0.25, "#FECACA"],
            [0.5, "#FFFFFF"], [0.75, "#99F6E4"], [1.0, "#0F766E"]
        ],
        colorbar=dict(title="Correlation", tickformat=".1f"),
        text=np.round(corr.to_numpy(), 2),
        texttemplate="%{text:.2f}",
        hovertemplate=(
            "Asset X: %{x}<br>Asset Y: %{y}<br>"
            "Correlation: %{z:.3f}<extra></extra>"
        )
    ))
    apply_layout(
        fig, "Asset Correlation & Diversification",
        "Correlations are estimated from the available daily return sample; lower correlation can improve diversification."
    )
    fig.update_xaxes(title="Asset", side="bottom")
    fig.update_yaxes(title="Asset", autorange="reversed")
    return fig


def html_kpi(title, value, note, color):
    return (
        '<div class="kpi">'
        f'<div class="kpi-label">{title}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-note">{note}</div>'
        '</div>'
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    frontier = load_frontier()
    weights_map = load_weights()

    asset_returns = returns[ASSETS]
    mean_returns = asset_returns.mean().to_numpy() * TRADING_DAYS
    covariance = asset_returns.cov().to_numpy() * TRADING_DAYS

    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Estimated covariance matrix is not positive semidefinite.")

    strategy_returns = pd.DataFrame(index=returns.index)
    for strategy, weights in weights_map.items():
        strategy_returns[strategy] = asset_returns.to_numpy() @ weights

    frontier_fig, optimal = make_frontier_chart(
        returns, frontier, weights_map, mean_returns, covariance
    )
    allocation_fig = make_allocation_chart(weights_map)
    growth_fig = make_growth_chart(returns, strategy_returns)
    drawdown_fig = make_drawdown_chart(strategy_returns, returns)
    rolling_fig = make_rolling_vol_chart(strategy_returns, returns)
    correlation_fig = make_correlation_chart(returns)

    gmv_ret, gmv_vol, gmv_sharpe = optimal["GMV"]
    max_ret, max_vol, max_sharpe = optimal["Maximum Sharpe"]

    stats = {
        s: performance_stats(strategy_returns[s])
        for s in STRATEGIES
    }
    benchmark_stats = performance_stats(returns[BENCHMARK])

    kpis = "".join([
        html_kpi("GMV · Expected Return", f"{gmv_ret:.2%}",
                 f"Volatility {gmv_vol:.2%} · Sharpe {gmv_sharpe:.3f}", "#7C3AED"),
        html_kpi("Maximum-Sharpe · Expected Return", f"{max_ret:.2%}",
                 f"Volatility {max_vol:.2%} · Sharpe {max_sharpe:.3f}", "#047857"),
        html_kpi("Sample Period", f"{returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}",
                 f"{len(returns):,} daily observations", "#2563EB"),
        html_kpi("Risk-Free Assumption", f"{RF:.2%}",
                 f"Maximum weight per asset: {CAP:.0%}", "#D97706")
    ])

    sections = [
        ("01", "Efficient Frontier & Optimal Portfolios",
         "Explore feasible portfolios, the efficient frontier, six individual assets, four strategies, the GMV portfolio and the constrained Maximum-Sharpe portfolio.",
         frontier_fig),
        ("02", "Strategic Asset Allocation",
         "Compare how each strategy distributes capital across the six investable assets.",
         allocation_fig),
        ("03", "Historical Portfolio Performance",
         "Compare historical growth against the S&P 500 benchmark. These are historical estimates, not guaranteed future returns.",
         growth_fig),
        ("04", "Drawdown & Downside Risk",
         "Compare losses from previous wealth peaks to assess historical downside experience.",
         drawdown_fig),
        ("05", "Rolling Volatility",
         "Track how realized risk changed through time using a 126-trading-day rolling window.",
         rolling_fig),
        ("06", "Asset Correlation & Diversification",
         "Inspect historical linear correlations among the six assets.",
         correlation_fig)
    ]

    config = {
        "responsive": True,
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"]
    }

    # One self-contained HTML page, with six independent interactive figures.
    chart_blocks = []
    for i, (number, title, description, fig) in enumerate(sections):
        chart_div = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs=True if i == 0 else False,
            config=config,
            default_width="100%"
        )
        chart_blocks.append(
            f'<section class="chart-section">'
            f'<div class="section-heading"><span>{number}</span>'
            f'<div><h2>{title}</h2><p>{description}</p></div></div>'
            f'{chart_div}</section>'
        )

    metrics_rows = []
    for strategy in STRATEGIES:
        s = stats[strategy]
        metrics_rows.append(
            f"<tr><td>{strategy}</td>"
            f"<td>{s['CAGR']:.2%}</td>"
            f"<td>{s['Annualized Volatility']:.2%}</td>"
            f"<td>{s['Sharpe Ratio']:.3f}</td>"
            f"<td>{s['Sortino Ratio']:.3f}</td>"
            f"<td>{s['Maximum Drawdown']:.2%}</td>"
            f"<td>{s['Calmar Ratio']:.3f}</td></tr>"
        )
    metrics_rows.append(
        f"<tr><td>S&P 500 (^GSPC)</td>"
        f"<td>{benchmark_stats['CAGR']:.2%}</td>"
        f"<td>{benchmark_stats['Annualized Volatility']:.2%}</td>"
        f"<td>{benchmark_stats['Sharpe Ratio']:.3f}</td>"
        f"<td>{benchmark_stats['Sortino Ratio']:.3f}</td>"
        f"<td>{benchmark_stats['Maximum Drawdown']:.2%}</td>"
        f"<td>{benchmark_stats['Calmar Ratio']:.3f}</td></tr>"
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Efficient Frontier Optimizer | Amit Singh</title>
<style>
:root {{
  --ink: #243447; --muted: #64748B; --line: #E5EAF1;
  --surface: #FFFFFF; --background: #F7F9FC;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--background); color: var(--ink);
  font-family: Inter, "Segoe UI", Arial, sans-serif; line-height: 1.55;
}}
.hero {{
  padding: 36px max(22px, calc((100vw - 1240px)/2));
  background: linear-gradient(120deg, #FFFFFF, #EEF7F6);
  border-bottom: 1px solid var(--line);
}}
.eyebrow {{ color: #0F766E; font-size: 12px; font-weight: 800;
  letter-spacing: .13em; text-transform: uppercase; }}
h1 {{ margin: 8px 0; font-size: clamp(27px, 4vw, 40px); line-height: 1.15; }}
.hero p {{ color: var(--muted); max-width: 900px; margin-bottom: 0; }}
.container {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px; margin-bottom: 24px; }}
.kpi {{ background: white; border: 1px solid var(--line); border-radius: 14px;
  padding: 18px; box-shadow: 0 3px 12px rgba(36,52,71,.035); min-width: 0; }}
.kpi-label {{ color: var(--muted); font-size: 12px; font-weight: 700; }}
.kpi-value {{ font-size: clamp(19px, 2vw, 27px); font-weight: 800;
  margin: 8px 0 4px; overflow-wrap: anywhere; }}
.kpi-note {{ color: var(--muted); font-size: 12px; }}
.chart-section {{ background: white; border: 1px solid var(--line);
  border-radius: 16px; padding: 18px 20px 12px; margin: 20px 0;
  box-shadow: 0 4px 16px rgba(36,52,71,.035); overflow: hidden; }}
.section-heading {{ display: flex; gap: 14px; align-items: flex-start;
  padding: 4px 4px 0; }}
.section-heading > span {{ display: inline-flex; flex: 0 0 38px; height: 38px;
  align-items: center; justify-content: center; border-radius: 10px;
  background: #E8F5F2; color: #0F766E; font-weight: 800; }}
h2 {{ font-size: 21px; margin: 0 0 4px; }}
.section-heading p {{ color: var(--muted); margin: 0 0 8px; font-size: 13px; }}
.table-wrap {{ overflow-x: auto; background: white; border: 1px solid var(--line);
  border-radius: 14px; margin-top: 22px; }}
table {{ border-collapse: collapse; width: 100%; min-width: 780px; }}
th, td {{ text-align: right; padding: 12px 14px; border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #F1F5F9; color: #475569; font-size: 12px; }}
td {{ font-size: 13px; }}
.note {{ color: var(--muted); font-size: 12px; margin: 14px 0 26px; }}
footer {{ text-align: center; color: var(--muted); padding: 22px; font-size: 12px; }}
@media (max-width: 850px) {{
  .kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .container {{ padding: 12px; }}
  .chart-section {{ padding: 12px 8px 6px; }}
}}
@media (max-width: 480px) {{
  .kpi-grid {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<header class="hero">
  <div class="eyebrow">Quantitative Portfolio Research</div>
  <h1>Efficient Frontier Optimizer</h1>
  <p>Interactive portfolio optimization and historical risk analysis.
  Six independent Plotly charts, one self-contained HTML report.
  Prepared by Amit Singh (nasaamit005).</p>
</header>
<main class="container">
  <div class="kpi-grid">{kpis}</div>
  {''.join(chart_blocks)}
  <section class="chart-section">
    <div class="section-heading"><span>07</span>
      <div><h2>Historical Performance Summary</h2>
      <p>Descriptive metrics calculated from the observed daily returns. They are not the optimizer's expected-return estimates.</p></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Strategy</th><th>CAGR</th><th>Annualized Volatility</th>
        <th>Sharpe</th><th>Sortino</th><th>Max Drawdown</th><th>Calmar</th></tr></thead>
        <tbody>{''.join(metrics_rows)}</tbody>
      </table>
    </div>
  </section>
  <p class="note">
    Methodology: expected returns and covariance are estimated from the available daily-return sample
    and annualized using 252 trading days. The 2% risk-free rate is an assumption.
    The Maximum-Sharpe solution is subject to the optimizer's constraints; its reference line
    should not be interpreted as proof of an unconstrained tangency portfolio.
    Simulated feasible points are illustrative samples of the constrained weight space, not an exhaustive enumeration.
  </p>
</main>
<footer>Efficient Frontier Optimizer · Interactive HTML research dashboard</footer>
</body>
</html>
"""

    OUTPUT.write_text(page, encoding="utf-8")

    print("\nDASHBOARD CREATED SUCCESSFULLY")
    print(f"File: {OUTPUT}")
    print(f"Observations: {len(returns):,}")
    print(f"Period: {returns.index.min():%Y-%m-%d} to {returns.index.max():%Y-%m-%d}")
    print("Independent charts: 6")
    print("HTML files created by this script: 1")
    print("\nOPTIMAL PORTFOLIO ESTIMATES")
    print(f"GMV:          Return {gmv_ret:.2%} | Volatility {gmv_vol:.2%} | Sharpe {gmv_sharpe:.3f}")
    print(f"Maximum Sharpe: Return {max_ret:.2%} | Volatility {max_vol:.2%} | Sharpe {max_sharpe:.3f}")
    print("\nHISTORICAL PERFORMANCE")
    print(pd.DataFrame(stats).T.round(4).to_string())


if __name__ == "__main__":
    main()
