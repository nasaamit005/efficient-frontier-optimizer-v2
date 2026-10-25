
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "BND", "^GSPC"]
START_DATE = "2016-01-01"


def download_prices() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading adjusted market prices: {', '.join(TICKERS)}")

    data = yf.download(
        TICKERS,
        start=START_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise RuntimeError(
            "No market data downloaded. Check your internet connection "
            "and try again."
        )

    # With auto_adjust=True, Close represents adjusted closing prices.
    prices = data["Close"].copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.reindex(columns=TICKERS)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    # Preserve the original downloaded data for auditability.
    raw_path = RAW_DIR / "adjusted_prices_download.csv"
    prices.to_csv(raw_path)

    # Report missing values before removing incomplete observations.
    missing = prices.isna().sum()
    print("\nMissing values by asset:")
    print(missing.to_string())

    # Drop dates where any asset is unavailable so all assets share
    # the same observation dates for portfolio optimization.
    clean_prices = prices.dropna(how="any")

    if clean_prices.empty:
        raise RuntimeError(
            "No complete observations remain after data cleaning."
        )

    if (clean_prices <= 0).any().any():
        raise ValueError("Non-positive prices detected.")

    returns = clean_prices.pct_change(fill_method=None).dropna()

    clean_path = PROCESSED_DIR / "adjusted_prices_clean.csv"
    returns_path = PROCESSED_DIR / "daily_returns.csv"

    clean_prices.to_csv(clean_path)
    returns.to_csv(returns_path)

    print("\nDATA DOWNLOAD COMPLETE")
    print(f"Assets: {', '.join(clean_prices.columns)}")
    print(f"Price observations: {len(clean_prices):,}")
    print(f"Return observations: {len(returns):,}")
    print(f"Start date: {clean_prices.index.min().date()}")
    print(f"End date: {clean_prices.index.max().date()}")
    print(f"\nRaw data: {raw_path}")
    print(f"Clean prices: {clean_path}")
    print(f"Daily returns: {returns_path}")

    return returns


if __name__ == "__main__":
    download_prices()
