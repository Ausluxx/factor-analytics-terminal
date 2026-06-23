from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger("factor_engine")

BENCHMARK_MKT = "^NSEI"
TRADING_DAYS_PER_YEAR = 252
RF_PROXY_TICKER = "^IRX"


@dataclass
class Diagnostics:
    notes: list[str] = field(default_factory=list)
    dropped_tickers: dict[str, str] = field(default_factory=dict)
    data_gaps: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def drop(self, ticker: str, reason: str):
        self.dropped_tickers[ticker] = reason
        logger.warning("Dropping %s: %s", ticker, reason)

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(msg)

    def note(self, msg: str):
        self.notes.append(msg)
        logger.info(msg)

    def as_dataframe(self) -> pd.DataFrame:
        rows = [{"Ticker": t, "Reason": r, "Type": "Dropped"} for t, r in self.dropped_tickers.items()]
        rows += [{"Ticker": t, "Reason": f"{n} missing trading days (NaN, not filled)", "Type": "Data Gap"}
                 for t, n in self.data_gaps.items()]
        return pd.DataFrame(rows)


def load_current_nifty50_universe() -> list[str]:
    return [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
        "SBIN.NS", "ITC.NS", "LT.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "MARUTI.NS",
        "HCLTECH.NS", "SUNPHARMA.NS", "ADANIENT.NS", "TMPV.NS", "AXISBANK.NS", "NTPC.NS",
        "TITAN.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "M&M.NS", "TATASTEEL.NS", "ADANIPORTS.NS",
        "ASIANPAINT.NS", "BAJAJ-AUTO.NS", "COALINDIA.NS", "INDUSINDBK.NS", "JIOFIN.NS",
        "HINDALCO.NS", "GRASIM.NS", "BRITANNIA.NS", "BPCL.NS", "DRREDDY.NS", "CIPLA.NS",
        "EICHERMOT.NS", "JSWSTEEL.NS", "NESTLEIND.NS", "TECHM.NS", "WIPRO.NS", "APOLLOHOSP.NS",
        "TATACONSUM.NS", "SHRIRAMFIN.NS", "BAJAJFINSV.NS", "BEL.NS", "TRENT.NS", "HAL.NS",
        "ONGC.NS", "DIVISLAB.NS",
    ]


def fetch_terminal_data(
    tickers: list[str], start: datetime.date, end: datetime.date, diag: Diagnostics
) -> pd.DataFrame:
    import yfinance as yf

    all_symbols = sorted(set(tickers) | {BENCHMARK_MKT})
    try:
        downloaded = yf.download(
            all_symbols, start=start, end=end, auto_adjust=True, progress=False
        )
    except (ConnectionError, TimeoutError) as e:
        diag.warn(f"Network error contacting data provider: {e}")
        return pd.DataFrame()
    except Exception as e:
        diag.warn(f"Unexpected error during download: {type(e).__name__}: {e}")
        return pd.DataFrame()

    if downloaded.empty or "Close" not in downloaded:
        diag.warn("Data provider returned an empty payload for the requested symbols/date range.")
        return pd.DataFrame()

    prices = downloaded["Close"]

    fully_missing = prices.columns[prices.isna().all()].tolist()
    for t in fully_missing:
        diag.drop(t, "No price data returned for the requested window (bad ticker, delisted, or no trading history).")
    prices = prices.drop(columns=fully_missing)

    if BENCHMARK_MKT not in prices.columns:
        diag.warn(f"Benchmark {BENCHMARK_MKT} returned no usable data for this window — cannot construct MKT_RF. Aborting.")
        return pd.DataFrame()

    expected_days = len(prices.index)
    for t in list(prices.columns):
        missing = int(prices[t].isna().sum())
        if missing > 0:
            diag.data_gaps[t] = missing
        if t == BENCHMARK_MKT:
            continue
        if expected_days > 0 and missing / expected_days > 0.10:
            diag.drop(t, f"{missing}/{expected_days} trading days missing (>10% gap) — excluded to avoid distorting size/value baskets and volatility.")
            prices = prices.drop(columns=[t])

    return prices


def fetch_fundamentals(tickers: list[str], diag: Diagnostics) -> pd.DataFrame:
    import yfinance as yf

    records = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            shares = info.get("sharesOutstanding")
            book_value_per_share = info.get("bookValue")
            if shares is None or book_value_per_share is None or book_value_per_share <= 0:
                diag.drop(t, "Missing shares outstanding or book value per share from data provider — cannot compute market cap or book-to-market.")
                continue
            records[t] = {"shares_outstanding": shares, "book_value_per_share": book_value_per_share}
        except (KeyError, ValueError) as e:
            diag.drop(t, f"Malformed fundamentals payload: {e}")
        except Exception as e:
            diag.drop(t, f"Unexpected error fetching fundamentals: {type(e).__name__}: {e}")

    if not records:
        diag.warn("No usable fundamentals retrieved for any ticker — SMB/HML cannot be constructed.")
    return pd.DataFrame(records).T


def fetch_risk_free_series(
    index: pd.DatetimeIndex, manual_annual_rate: Optional[float], diag: Diagnostics
) -> pd.Series:
    if manual_annual_rate is None:
        manual_annual_rate = 0.065
        diag.warn("No risk-free rate supplied; defaulting to 6.5% annualized.")

    diag.note(
        f"Risk-free rate held constant at {manual_annual_rate:.3%} (annualized) across the "
        "full window — no free, reliable historical Indian T-Bill series was available. "
        "Supply a real historical series here if available."
    )
    daily_rate = manual_annual_rate / TRADING_DAYS_PER_YEAR
    return pd.Series(daily_rate, index=index, name="RF")


def build_fama_french_factors(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    rf_series: pd.Series,
    diag: Diagnostics,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices_df.pct_change(fill_method=None)
    stock_cols = [c for c in returns.columns if c != BENCHMARK_MKT]
    returns = returns.dropna(subset=[BENCHMARK_MKT])

    common_tickers = [t for t in stock_cols if t in fundamentals_df.index]
    missing_fundamentals = set(stock_cols) - set(common_tickers)
    for t in missing_fundamentals:
        diag.drop(
            t,
            "No fundamentals available — excluded from SMB/HML basket construction. "
            "Factor betas are still estimated against the resulting factors.",
        )

    if len(common_tickers) < 6:
        diag.warn(f"Only {len(common_tickers)} tickers have usable fundamentals — SMB/HML may be unstable with this few names per basket.")

    factors = pd.DataFrame(index=returns.index)
    rf_aligned = rf_series.reindex(returns.index).ffill()
    factors["MKT_RF"] = returns[BENCHMARK_MKT] - rf_aligned
    factors["RF"] = rf_aligned

    years = returns.index.year
    smb_vals = pd.Series(index=returns.index, dtype=float)
    hml_vals = pd.Series(index=returns.index, dtype=float)

    for yr in sorted(set(years)):
        year_mask = years == yr
        year_dates = returns.index[year_mask]
        if len(year_dates) == 0:
            continue
        anchor_date = year_dates[0]

        price_at_anchor = prices_df.loc[anchor_date, common_tickers]
        shares = fundamentals_df.loc[common_tickers, "shares_outstanding"].astype(float)
        market_cap = price_at_anchor * shares

        book_per_share = fundamentals_df.loc[common_tickers, "book_value_per_share"].astype(float)
        book_to_market = book_per_share / price_at_anchor

        valid = market_cap.notna() & book_to_market.notna() & (price_at_anchor > 0)
        market_cap = market_cap[valid]
        book_to_market = book_to_market[valid]

        if len(market_cap) < 6:
            diag.warn(f"Year {yr}: fewer than 6 valid names for SMB/HML basket construction; factor values for this year may be noisy.")
            continue

        size_median = market_cap.median()
        small = market_cap[market_cap <= size_median].index
        big = market_cap[market_cap > size_median].index

        btm_30 = book_to_market.quantile(0.30)
        btm_70 = book_to_market.quantile(0.70)
        growth = book_to_market[book_to_market <= btm_30].index
        value = book_to_market[book_to_market >= btm_70].index

        sv = [t for t in small if t in value]
        sg = [t for t in small if t in growth]
        bv = [t for t in big if t in value]
        bg = [t for t in big if t in growth]

        year_returns = returns.loc[year_mask, common_tickers]

        def basket_mean(names):
            if len(names) == 0:
                return pd.Series(np.nan, index=year_returns.index)
            return year_returns[names].mean(axis=1)

        r_sv, r_sg, r_bv, r_bg = basket_mean(sv), basket_mean(sg), basket_mean(bv), basket_mean(bg)

        r_sv = r_sv.where(r_sv.notna(), basket_mean(small))
        r_sg = r_sg.where(r_sg.notna(), basket_mean(small))
        r_bv = r_bv.where(r_bv.notna(), basket_mean(big))
        r_bg = r_bg.where(r_bg.notna(), basket_mean(big))

        smb_vals.loc[year_dates] = ((r_sv + r_sg) / 2 - (r_bv + r_bg) / 2).values
        hml_vals.loc[year_dates] = ((r_sv + r_bv) / 2 - (r_sg + r_bg) / 2).values

    factors["SMB"] = smb_vals
    factors["HML"] = hml_vals
    factors = factors.dropna(subset=["MKT_RF", "SMB", "HML"])
    aligned_returns = returns.loc[factors.index, stock_cols]

    return factors, aligned_returns


MIN_OBS_FOR_REGRESSION = 60


def run_factor_regressions(returns_df: pd.DataFrame, factors_df: pd.DataFrame, diag: Diagnostics) -> pd.DataFrame:
    results = {}
    X = sm.add_constant(factors_df[["MKT_RF", "SMB", "HML"]])

    for stock in returns_df.columns:
        y = returns_df[stock] - factors_df["RF"]
        common = y.dropna().index.intersection(X.index)
        n_obs = len(common)

        if n_obs < MIN_OBS_FOR_REGRESSION:
            diag.drop(stock, f"Only {n_obs} usable observations (< minimum {MIN_OBS_FOR_REGRESSION}) — regression skipped.")
            continue

        try:
            maxlags = max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))
            model = sm.OLS(y.loc[common], X.loc[common]).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        except np.linalg.LinAlgError as e:
            diag.drop(stock, f"Singular design matrix: {e}")
            continue
        except ValueError as e:
            diag.drop(stock, f"Regression input error: {e}")
            continue

        resid_returns = y.loc[common]
        naive_vol = resid_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100
        ewma_vol = _ewma_annualized_vol(resid_returns) * 100

        results[stock] = {
            "Alpha (Ann %)": model.params["const"] * TRADING_DAYS_PER_YEAR * 100,
            "Beta Market": model.params["MKT_RF"],
            "Beta Size (SMB)": model.params["SMB"],
            "Beta Value (HML)": model.params["HML"],
            "t-stat Alpha": model.tvalues["const"],
            "R-Squared": model.rsquared,
            "Volatility (Ann %, i.i.d.)": naive_vol,
            "Volatility (Ann %, EWMA)": ewma_vol,
            "N (obs)": n_obs,
            "DoF": n_obs - 4,
            "HAC Lags": maxlags,
        }

    if not results:
        diag.warn("No stock cleared the minimum-sample regression threshold — check date range and universe.")
    return pd.DataFrame(results).T


def _ewma_annualized_vol(returns: pd.Series, lam: float = 0.94) -> float:
    r = returns.dropna().values
    if len(r) < 2:
        return float("nan")
    var = r[0] ** 2
    for x in r[1:]:
        var = lam * var + (1 - lam) * x ** 2
    return float(np.sqrt(var * TRADING_DAYS_PER_YEAR))


def calculate_rolling_exposures(
    stock_series: pd.Series, factors_df: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    X = sm.add_constant(factors_df[["MKT_RF", "SMB", "HML"]])
    common = stock_series.dropna().index.intersection(X.index)
    y = stock_series.loc[common]
    X_clean = X.loc[common]
    records = []

    if window <= 4:
        raise ValueError("Rolling window must exceed the number of regression parameters (4).")

    for i in range(window, len(y)):
        y_slice = y.iloc[i - window:i] - factors_df["RF"].reindex(y.index).iloc[i - window:i].values
        X_slice = X_clean.iloc[i - window:i]
        try:
            model = sm.OLS(y_slice, X_slice).fit()
        except (np.linalg.LinAlgError, ValueError):
            continue
        records.append({
            "Date": y.index[i],
            "Market Factor": model.params["MKT_RF"],
            "Size Factor (SMB)": model.params["SMB"],
            "Value Factor (HML)": model.params["HML"],
        })
    return pd.DataFrame(records).set_index("Date") if records else pd.DataFrame(
        columns=["Market Factor", "Size Factor (SMB)", "Value Factor (HML)"]
    )
