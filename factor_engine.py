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


# ── Diagnostics ───────────────────────────────────────────────────────────────
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
        rows = [
            {"Ticker": t, "Reason": r, "Type": "Dropped"}
            for t, r in self.dropped_tickers.items()
        ]
        rows += [
            {"Ticker": t, "Reason": f"{n} missing trading days (NaN, not filled)", "Type": "Data Gap"}
            for t, n in self.data_gaps.items()
        ]
        return pd.DataFrame(rows)


# ── Universe ──────────────────────────────────────────────────────────────────
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
        "ONGC.NS", "DIVISLAB.NS", "KOTAKBANK.NS",
    ]


# ── Data fetching ─────────────────────────────────────────────────────────────
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

    # Flatten MultiIndex columns — yfinance v0.2+ sometimes returns one
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    fully_missing = prices.columns[prices.isna().all()].tolist()
    for t in fully_missing:
        diag.drop(t, "No price data returned (bad ticker, delisted, or no trading history).")
    prices = prices.drop(columns=fully_missing)

    if BENCHMARK_MKT not in prices.columns:
        diag.warn(f"Benchmark {BENCHMARK_MKT} returned no usable data — cannot construct MKT_RF.")
        return pd.DataFrame()

    expected_days = len(prices.index)
    for t in list(prices.columns):
        missing = int(prices[t].isna().sum())
        if missing > 0:
            diag.data_gaps[t] = missing
        if t == BENCHMARK_MKT:
            continue
        if expected_days > 0 and missing / expected_days > 0.10:
            diag.drop(t, f"{missing}/{expected_days} trading days missing (>10% gap).")
            prices = prices.drop(columns=[t])

    return prices


def _extract_fundamentals_from_info(info: dict) -> tuple[float | None, float | None]:
    """Try every known yfinance field name for shares and book value."""
    shares = (
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
        or info.get("floatShares")
    )
    book = (
        info.get("bookValue")
        or info.get("bookValuePerShare")
        or info.get("priceToBook") and info.get("previousClose")
        # priceToBook = price/book → book = price/priceToBook
    )
    # Fallback: derive book-per-share from priceToBook ratio if bookValue missing
    if book is None or book <= 0:
        ptb = info.get("priceToBook")
        price = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if ptb and price and ptb > 0 and price > 0:
            book = price / ptb
    return (
        float(shares) if shares and shares > 0 else None,
        float(book) if book and book > 0 else None,
    )


def _get_fundamentals_fast_info(ticker_obj) -> tuple[float | None, float | None]:
    """Try yfinance fast_info — more reliable than .info in recent versions."""
    try:
        fi = ticker_obj.fast_info
        shares = getattr(fi, "shares", None)
        # fast_info has no book value — return shares only
        return (float(shares) if shares and shares > 0 else None, None)
    except Exception:
        return (None, None)


def _get_book_from_balance_sheet(ticker_obj) -> float | None:
    """Derive book value per share from the balance sheet as last resort."""
    try:
        bs = ticker_obj.quarterly_balance_sheet
        if bs is None or bs.empty:
            bs = ticker_obj.balance_sheet
        if bs is None or bs.empty:
            return None
        # Stockholders equity rows vary by yfinance version
        for row in ["Stockholders Equity", "Total Stockholder Equity",
                    "Common Stock Equity", "Total Equity Gross Minority Interest"]:
            if row in bs.index:
                equity = float(bs.loc[row].iloc[0])
                if equity > 0:
                    return equity  # total equity — divided by shares later
        return None
    except Exception:
        return None


def fetch_fundamentals(tickers: list[str], diag: Diagnostics) -> pd.DataFrame:
    import yfinance as yf

    records: dict[str, dict] = {}

    for t in tickers:
        shares: float | None = None
        book: float | None = None

        try:
            tk = yf.Ticker(t)

            # ── Method 1: .info dict ───────────────────────────────────────────
            try:
                info = tk.info
                if info and len(info) > 5:          # non-empty response
                    shares, book = _extract_fundamentals_from_info(info)
            except Exception:
                info = {}

            # ── Method 2: fast_info for shares (if Method 1 missed shares) ────
            if shares is None:
                shares_fi, _ = _get_fundamentals_fast_info(tk)
                if shares_fi:
                    shares = shares_fi

            # ── Method 3: balance sheet for book equity (if still missing) ────
            if book is None or book <= 0:
                total_equity = _get_book_from_balance_sheet(tk)
                if total_equity and shares and shares > 0:
                    book = total_equity / shares   # converts total equity → per-share

            # ── Validate ───────────────────────────────────────────────────────
            if shares is None or book is None or book <= 0:
                diag.drop(
                    t,
                    f"Could not retrieve shares outstanding or book value after 3 methods "
                    f"(shares={'None' if shares is None else f'{shares:.0f}'}, "
                    f"book={'None' if book is None else f'{book:.2f}'})."
                )
                continue

            records[t] = {
                "shares_outstanding": float(shares),
                "book_value_per_share": float(book),
            }

        except Exception as e:
            diag.drop(t, f"Unexpected error fetching fundamentals: {type(e).__name__}: {e}")

    if not records:
        diag.warn("No usable fundamentals retrieved — SMB/HML cannot be constructed.")
        return pd.DataFrame(columns=["shares_outstanding", "book_value_per_share"])

    df = pd.DataFrame.from_dict(records, orient="index")
    df["shares_outstanding"] = df["shares_outstanding"].astype(float)
    df["book_value_per_share"] = df["book_value_per_share"].astype(float)
    return df


def fetch_risk_free_series(
    index: pd.DatetimeIndex, manual_annual_rate: Optional[float], diag: Diagnostics
) -> pd.Series:
    if manual_annual_rate is None:
        manual_annual_rate = 0.065
        diag.warn("No risk-free rate supplied; defaulting to 6.5% annualised.")

    diag.note(
        f"Risk-free rate held constant at {manual_annual_rate:.3%} (annualised) across the full "
        "window — no free reliable historical Indian T-Bill series available."
    )
    daily_rate = manual_annual_rate / TRADING_DAYS_PER_YEAR
    return pd.Series(daily_rate, index=index, name="RF")


# ── Factor construction ───────────────────────────────────────────────────────
def build_fama_french_factors(
    prices_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    rf_series: pd.Series,
    diag: Diagnostics,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # pct_change with fill_method=None preserves NaN — never fabricates zero returns
    returns = prices_df.pct_change(fill_method=None)
    stock_cols = [c for c in returns.columns if c != BENCHMARK_MKT]

    # Drop only rows where benchmark is NaN; stock NaNs are preserved
    returns = returns.dropna(subset=[BENCHMARK_MKT])

    # Match tickers that have both prices and fundamentals
    if fundamentals_df.empty:
        diag.warn("Fundamentals DataFrame is empty — SMB/HML cannot be constructed.")
        common_tickers: list[str] = []
    else:
        common_tickers = [t for t in stock_cols if t in fundamentals_df.index]

    for t in sorted(set(stock_cols) - set(common_tickers)):
        diag.drop(t, "No fundamentals — excluded from SMB/HML basket construction.")

    if len(common_tickers) < 6:
        diag.warn(f"Only {len(common_tickers)} tickers have usable fundamentals — SMB/HML baskets will be noisy.")

    # Factor scaffold
    factors = pd.DataFrame(index=returns.index)
    rf_aligned = rf_series.reindex(returns.index).ffill()
    factors["MKT_RF"] = returns[BENCHMARK_MKT] - rf_aligned
    factors["RF"] = rf_aligned

    smb_vals = pd.Series(np.nan, index=returns.index, dtype=float)
    hml_vals = pd.Series(np.nan, index=returns.index, dtype=float)

    years = returns.index.year

    for yr in sorted(set(years)):
        year_mask = years == yr
        year_dates = returns.index[year_mask]

        if len(year_dates) == 0 or len(common_tickers) == 0:
            continue

        anchor_date = year_dates[0]
        price_at_anchor = prices_df.loc[anchor_date, common_tickers].astype(float)
        shares = fundamentals_df.loc[common_tickers, "shares_outstanding"].astype(float)
        book_per_share = fundamentals_df.loc[common_tickers, "book_value_per_share"].astype(float)

        market_cap = price_at_anchor * shares
        book_to_market = book_per_share / price_at_anchor

        # Strict validity: non-null, finite, positive price, finite B/M
        valid = (
            market_cap.notna()
            & book_to_market.notna()
            & (price_at_anchor > 0)
            & np.isfinite(market_cap)
            & np.isfinite(book_to_market)
        )
        market_cap = market_cap[valid]
        book_to_market = book_to_market[valid]

        if len(market_cap) < 6:
            diag.warn(f"Year {yr}: only {len(market_cap)} valid names — skipping SMB/HML.")
            continue

        # Size: median split
        size_median = market_cap.median()
        small = market_cap[market_cap <= size_median].index.tolist()
        big   = market_cap[market_cap >  size_median].index.tolist()

        # Value: 30th / 70th percentile B/M split; middle 40% excluded from basket definition
        btm_30 = book_to_market.quantile(0.30)
        btm_70 = book_to_market.quantile(0.70)
        growth = book_to_market[book_to_market <= btm_30].index.tolist()
        value  = book_to_market[book_to_market >= btm_70].index.tolist()

        # Six portfolios
        sv = [t for t in small if t in value]
        sg = [t for t in small if t in growth]
        bv = [t for t in big   if t in value]
        bg = [t for t in big   if t in growth]

        year_returns = returns.loc[year_mask, common_tickers]

        # Explicit parameter avoids closure-over-loop-variable risk
        def basket_mean(names: list[str], yr_rets: pd.DataFrame) -> pd.Series:
            if not names:
                return pd.Series(np.nan, index=yr_rets.index)
            return yr_rets[names].mean(axis=1)

        r_sv = basket_mean(sv, year_returns)
        r_sg = basket_mean(sg, year_returns)
        r_bv = basket_mean(bv, year_returns)
        r_bg = basket_mean(bg, year_returns)

        # Fallback: if a sub-basket is entirely empty, use the broader size bucket
        r_sv = r_sv.where(r_sv.notna(), basket_mean(small, year_returns))
        r_sg = r_sg.where(r_sg.notna(), basket_mean(small, year_returns))
        r_bv = r_bv.where(r_bv.notna(), basket_mean(big,   year_returns))
        r_bg = r_bg.where(r_bg.notna(), basket_mean(big,   year_returns))

        # SMB = small_avg − big_avg (pure size, orthogonal to value)
        # HML = value_avg − growth_avg (pure value, orthogonal to size)
        smb_year = (r_sv + r_sg) / 2.0 - (r_bv + r_bg) / 2.0
        hml_year = (r_sv + r_bv) / 2.0 - (r_sg + r_bg) / 2.0

        smb_vals.loc[year_dates] = smb_year.values
        hml_vals.loc[year_dates] = hml_year.values

    factors["SMB"] = smb_vals
    factors["HML"] = hml_vals

    # Keep only dates where all three factors are non-NaN
    factors = factors.dropna(subset=["MKT_RF", "SMB", "HML"])
    aligned_returns = returns.loc[factors.index, stock_cols]

    return factors, aligned_returns


# ── Regressions ───────────────────────────────────────────────────────────────
MIN_OBS_FOR_REGRESSION = 60


def run_factor_regressions(
    returns_df: pd.DataFrame, factors_df: pd.DataFrame, diag: Diagnostics
) -> pd.DataFrame:
    results: dict[str, dict] = {}
    X = sm.add_constant(factors_df[["MKT_RF", "SMB", "HML"]])

    for stock in returns_df.columns:
        y = returns_df[stock] - factors_df["RF"]
        # Intersect valid stock-return dates with valid factor dates
        common = y.dropna().index.intersection(X.dropna().index)
        n_obs = len(common)

        if n_obs < MIN_OBS_FOR_REGRESSION:
            diag.drop(stock, f"Only {n_obs} usable observations (< {MIN_OBS_FOR_REGRESSION}) — regression skipped.")
            continue

        try:
            # Newey-West lag: Andrews (1991) data-dependent rule
            maxlags = max(1, int(np.floor(4 * (n_obs / 100) ** (2 / 9))))
            model = sm.OLS(y.loc[common], X.loc[common]).fit(
                cov_type="HAC", cov_kwds={"maxlags": maxlags}
            )
        except np.linalg.LinAlgError as e:
            diag.drop(stock, f"Singular design matrix: {e}")
            continue
        except ValueError as e:
            diag.drop(stock, f"Regression input error: {e}")
            continue

        resid = y.loc[common]
        naive_vol = float(resid.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0
        ewma_vol  = _ewma_annualized_vol(resid) * 100.0

        # All values cast to Python scalars — prevents dtype leakage into DataFrame
        results[stock] = {
            "Alpha (Ann %)":              float(model.params["const"]) * TRADING_DAYS_PER_YEAR * 100.0,
            "Beta Market":                float(model.params["MKT_RF"]),
            "Beta Size (SMB)":            float(model.params["SMB"]),
            "Beta Value (HML)":           float(model.params["HML"]),
            "t-stat Alpha":               float(model.tvalues["const"]),
            "R-Squared":                  float(model.rsquared),
            "Volatility (Ann %, i.i.d.)": float(naive_vol),
            "Volatility (Ann %, EWMA)":   float(ewma_vol),
            "N (obs)":                    int(n_obs),
            "DoF":                        int(n_obs - 4),
            "HAC Lags":                   int(maxlags),
        }

    if not results:
        diag.warn("No stock cleared the minimum-sample regression threshold — check date range and universe.")
        return pd.DataFrame()

    return pd.DataFrame.from_dict(results, orient="index")


def _ewma_annualized_vol(returns: pd.Series, lam: float = 0.94) -> float:
    """EWMA conditional volatility (RiskMetrics, λ=0.94), annualised."""
    r = returns.dropna().values.astype(float)
    if len(r) < 2:
        return float("nan")
    var = r[0] * r[0]
    for x in r[1:]:
        var = lam * var + (1.0 - lam) * x * x
    return float(np.sqrt(var * TRADING_DAYS_PER_YEAR))


# ── Rolling exposures ─────────────────────────────────────────────────────────
def calculate_rolling_exposures(
    stock_series: pd.Series, factors_df: pd.DataFrame, window: int = 60
) -> pd.DataFrame:
    if window <= 4:
        raise ValueError("Rolling window must exceed the number of regression parameters (4).")

    X = sm.add_constant(factors_df[["MKT_RF", "SMB", "HML"]])

    # Align on dates where both stock returns and factor returns exist
    common_idx = stock_series.dropna().index.intersection(X.dropna().index)

    if len(common_idx) < window + 1:
        return pd.DataFrame(columns=["Market Factor", "Size Factor (SMB)", "Value Factor (HML)"])

    y = stock_series.loc[common_idx]
    X_clean = X.loc[common_idx]
    # Pre-align RF so slice indexing is always consistent
    rf_aligned = factors_df["RF"].reindex(common_idx)

    records = []
    for i in range(window, len(y)):
        y_slice = y.iloc[i - window:i].values - rf_aligned.iloc[i - window:i].values
        X_slice = X_clean.iloc[i - window:i]
        try:
            model = sm.OLS(y_slice, X_slice).fit()
        except (np.linalg.LinAlgError, ValueError):
            continue
        records.append({
            "Date":               y.index[i],
            "Market Factor":      float(model.params["MKT_RF"]),
            "Size Factor (SMB)":  float(model.params["SMB"]),
            "Value Factor (HML)": float(model.params["HML"]),
        })

    if not records:
        return pd.DataFrame(columns=["Market Factor", "Size Factor (SMB)", "Value Factor (HML)"])
    return pd.DataFrame(records).set_index("Date")
