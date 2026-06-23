import streamlit as st
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Factor Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0a0a0d;
        background-image: radial-gradient(circle at 50% 0%, #14141a 0%, #0a0a0d 60%);
    }

    .block-container { padding-top: 1.25rem; padding-bottom: 1.5rem; max-width: 97%; }

    h1, h2, h3, h4 { font-weight: 700 !important; letter-spacing: -0.02em; color: #f5f5f7; }
    p, span, label, div { color: #d4d4d8; }

    .terminal-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.4rem 0 1.1rem 0;
        border-bottom: 1px solid #1c1c22;
        margin-bottom: 1.25rem;
    }
    .terminal-header .logo-badge {
        width: 38px; height: 38px;
        border-radius: 9px;
        background: linear-gradient(135deg, #f87171, #ef4444);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .terminal-header .title-text {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        color: #f5f5f7;
        text-transform: uppercase;
    }

    .metric-card {
        position: relative;
        background-color: #111114;
        padding: 1.1rem 1.3rem 1.2rem 1.3rem;
        border-radius: 0.85rem;
        border: 1px solid #3f1d1d;
        box-shadow: 0 0 0 1px rgba(248,113,113,0.04), 0 8px 18px -8px rgba(0,0,0,0.5);
        height: 100%;
    }
    .metric-badge {
        position: absolute;
        top: 0.85rem; right: 0.95rem;
        width: 26px; height: 26px;
        border-radius: 7px;
        background: rgba(248,113,113,0.12);
        border: 1px solid rgba(248,113,113,0.35);
        color: #f87171;
        font-weight: 700;
        font-size: 0.85rem;
        display: flex; align-items: center; justify-content: center;
    }
    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9a9aa3;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #fca5a5;
        line-height: 1.1;
    }
    .metric-value.green { color: #6ee7b7; }
    .metric-sub {
        font-size: 0.72rem;
        color: #71717a;
        margin-top: 0.35rem;
        font-weight: 500;
    }

    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #8b8b93 !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: #f87171 !important; }
    div[data-baseweb="tab-highlight"] { background-color: #f87171 !important; height: 3px !important; }
    div[data-baseweb="tab-border"] { background-color: #1c1c22 !important; }
    div[data-baseweb="tab-list"] { gap: 1.8rem; }

    .panel-box {
        background-color: #0e0e11;
        border: 1px solid #1c1c22;
        border-radius: 0.85rem;
        padding: 1.1rem 1.2rem;
        height: 100%;
    }
    .panel-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #f5f5f7;
        margin-bottom: 0.85rem;
    }
    .status-box {
        background-color: #16161a;
        padding: 0.85rem 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #f87171;
        margin-bottom: 1rem;
        font-size: 0.88rem;
    }

    div[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700 !important; color: #fca5a5 !important; }
    div[data-testid="stMetricLabel"] > label { font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.05em; color: #9a9aa3 !important; }

    div[data-testid="stDataFrame"] { border: 1px solid #1c1c22; border-radius: 0.6rem; }

    .footer-bar {
        margin-top: 1.4rem;
        padding: 0.7rem 1.1rem;
        background-color: #0e0e11;
        border: 1px solid #1c1c22;
        border-radius: 0.6rem;
        font-size: 0.78rem;
        color: #8b8b93;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .footer-bar .ok { color: #6ee7b7; font-weight: 600; }
    .footer-bar .sep { color: #3a3a42; margin: 0 0.6rem; }

    section[data-testid="stSidebar"] { background-color: #0c0c0f; border-right: 1px solid #1c1c22; }
    </style>
""", unsafe_allow_html=True)

from factor_engine import (
    Diagnostics,
    load_current_nifty50_universe,
    fetch_terminal_data as _fetch_terminal_data,
    fetch_fundamentals as _fetch_fundamentals,
    fetch_risk_free_series,
    build_fama_french_factors,
    run_factor_regressions,
    calculate_rolling_exposures,
    BENCHMARK_MKT,
)

ACCENT = "#f87171"
ACCENT_SOFT = "#fca5a5"
GREEN = "#6ee7b7"
AMBER = "#fbbf24"


@st.cache_data(show_spinner="Downloading historical price data...")
def fetch_terminal_data_cached(tickers: list, start, end):
    local_diag = Diagnostics()
    prices = _fetch_terminal_data(tickers, start, end, local_diag)
    return prices, local_diag.dropped_tickers, local_diag.data_gaps, local_diag.warnings


@st.cache_data(show_spinner="Fetching shares outstanding and book value...")
def fetch_fundamentals_cached(tickers: tuple):
    local_diag = Diagnostics()
    fundamentals = _fetch_fundamentals(list(tickers), local_diag)
    return fundamentals, local_diag.dropped_tickers, local_diag.warnings


def merge_diag_payload(diag: Diagnostics, dropped: dict, gaps: dict = None, warnings_list: list = None):
    diag.dropped_tickers.update(dropped)
    if gaps:
        diag.data_gaps.update(gaps)
    if warnings_list:
        for w in warnings_list:
            if w not in diag.warnings:
                diag.warnings.append(w)


@st.cache_data
def load_current_nifty50_universe_cached():
    return load_current_nifty50_universe()


def render_metric_card(badge_num, label, value, sub=None, green=False):
    value_class = "metric-value green" if green else "metric-value"
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-badge">{badge_num}</div>
            <div class="metric-label">{label}</div>
            <div class="{value_class}">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


def style_plotly_dark(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#d4d4d8"),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    if height:
        fig.update_layout(height=height)
    return fig


with st.sidebar:
    st.markdown("## Controls")
    universe_selection = st.selectbox("Universe", ["Nifty 50", "Custom Tickers"])

    if universe_selection == "Custom Tickers":
        custom_input = st.text_area("Tickers (comma-separated)", value="RELIANCE.NS, TCS.NS, HDFCBANK.NS")
        active_universe = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    else:
        active_universe = load_current_nifty50_universe_cached()
        st.caption(
            "⚠️ Uses today's Nifty 50 members projected backward. Stocks added or removed "
            "mid-window are included or excluded for the entire period (survivorship bias). "
            "Universe-level averages are approximate."
        )

    st.markdown("---")
    st.markdown("### Factor Sorting")
    sort_attribute = st.selectbox(
        "Sort by",
        ["Alpha (Ann %)", "R-Squared", "Volatility (Ann %, EWMA)", "Volatility (Ann %, i.i.d.)", "Beta Market"]
    )

    st.markdown("---")
    lookback_years = st.slider("Lookback Window (Years)", 1, 5, 2)

    st.markdown("---")
    st.markdown("### Risk-Free Rate")
    rbi_rate = st.number_input("Annualized Rate", 0.0, 0.15, 0.065, step=0.005, format="%.4f")
    st.caption(
        "Held constant across the full window — no free daily Indian T-Bill series is available. "
        "Flagged as a known simplification in the Diagnostics tab."
    )

end_date = datetime.date.today()
start_date = end_date.replace(year=end_date.year - lookback_years)

st.markdown("""
    <div class="terminal-header">
        <div class="logo-badge">📈</div>
        <div class="title-text">Factor Analytics</div>
    </div>
""", unsafe_allow_html=True)

diag = Diagnostics()
raw_data, _dropped, _gaps, _warnings = fetch_terminal_data_cached(active_universe, start_date, end_date)
merge_diag_payload(diag, _dropped, _gaps, _warnings)

if raw_data.empty or len(raw_data.columns) <= 1:
    st.error("No data returned. Check your tickers, date range, or internet connection.")
    if diag.warnings:
        with st.expander("Diagnostics"):
            for w in diag.warnings:
                st.write(f"- {w}")
else:
    stock_tickers = [c for c in raw_data.columns if c != BENCHMARK_MKT]

    fundamentals, _f_dropped, _f_warnings = fetch_fundamentals_cached(tuple(sorted(stock_tickers)))
    merge_diag_payload(diag, _f_dropped, warnings_list=_f_warnings)

    rf_series = fetch_risk_free_series(raw_data.index, manual_annual_rate=rbi_rate, diag=diag)
    factors, returns = build_fama_french_factors(raw_data, fundamentals, rf_series, diag)

    if factors.empty or returns.empty:
        st.error("Factor construction produced no usable rows. Check the Diagnostics tab — likely cause is insufficient fundamentals coverage or too few names per annual basket.")
        with st.expander("Diagnostics", expanded=True):
            st.dataframe(diag.as_dataframe(), use_container_width=True)
            for w in diag.warnings:
                st.write(f"⚠️ {w}")
        st.stop()

    regression_matrix = run_factor_regressions(returns, factors, diag)

    if regression_matrix.empty:
        st.error("No stock cleared the minimum-sample regression threshold. Try a longer lookback window or a larger universe.")
        with st.expander("Diagnostics", expanded=True):
            st.dataframe(diag.as_dataframe(), use_container_width=True)
        st.stop()

    regression_matrix["Z_Score_Rank"] = (
        (regression_matrix[sort_attribute] - regression_matrix[sort_attribute].mean())
        / regression_matrix[sort_attribute].std()
    )
    sorted_matrix = regression_matrix.sort_values(by="Z_Score_Rank", ascending=False)

    st.markdown(f"Analysing **{len(stock_tickers)}** equities from **{start_date}** to **{end_date}**")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("1", "Equities Regressed", f"{len(sorted_matrix)} / {len(stock_tickers)}", sub=f"AS OF: {end_date}")
    with m2:
        render_metric_card("2", "Mean Cross-Sectional R²", f"{sorted_matrix['R-Squared'].mean():.0%}", green=True)
    with m3:
        render_metric_card("3", "Top Alpha", f"{sorted_matrix.index[0]}")
    with m4:
        n_issues = len(diag.dropped_tickers) + len(diag.warnings)
        render_metric_card("4", "Diagnostic Flags", f"{n_issues}", sub="dropped tickers + warnings", green=(n_issues == 0))

    st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)

    tab_screener, tab_heatmap, tab_decomposition, tab_diagnostics = st.tabs([
        "Screener",
        "Factor Heatmap",
        "Decomposition",
        "Diagnostics"
    ])

    with tab_screener:
        col_table, col_scatter = st.columns([1.3, 1], gap="medium")

        with col_table:
            st.markdown(
                f"<div class='panel-title'>Universe Ranking &nbsp;"
                f"<span style='color:#71717a;font-weight:500;font-size:0.85rem;'>Sorted by {sort_attribute} Z-Score</span></div>",
                unsafe_allow_html=True
            )
            display_df = sorted_matrix.copy()
            display_df["Alpha (Ann %)"] = display_df["Alpha (Ann %)"].map("{:+.2f}%".format)
            display_df["Beta Market"] = display_df["Beta Market"].map("{:.3f}".format)
            display_df["Beta Size (SMB)"] = display_df["Beta Size (SMB)"].map("{:+.3f}".format)
            display_df["Beta Value (HML)"] = display_df["Beta Value (HML)"].map("{:+.3f}".format)
            display_df["Volatility (Ann %, i.i.d.)"] = display_df["Volatility (Ann %, i.i.d.)"].map("{:.2f}%".format)
            display_df["Volatility (Ann %, EWMA)"] = display_df["Volatility (Ann %, EWMA)"].map("{:.2f}%".format)
            display_df["R-Squared"] = display_df["R-Squared"].map("{:.2%}".format)
            display_df["Z_Score_Rank"] = display_df["Z_Score_Rank"].map("{:+.2f}".format)
            display_df["t-stat Alpha"] = display_df["t-stat Alpha"].map("{:+.2f}".format)
            display_df["N (obs)"] = display_df["N (obs)"].map("{:.0f}".format)
            display_df["DoF"] = display_df["DoF"].map("{:.0f}".format)
            display_df["HAC Lags"] = display_df["HAC Lags"].map("{:.0f}".format)
            st.dataframe(display_df, use_container_width=True, height=480)
            st.caption(
                "EWMA volatility (λ=0.94) weights recent observations more heavily and responds "
                "to volatility clustering; i.i.d. volatility is the flat full-window stdev × √252. "
                "DoF = N − 4 (3 factors + intercept). HAC lags follow the Newey-West rule of thumb."
            )

        with col_scatter:
            st.markdown("<div class='panel-title'>Risk / Return</div>", unsafe_allow_html=True)
            scatter_fig = px.scatter(
                sorted_matrix.reset_index(),
                x="Volatility (Ann %, EWMA)",
                y="Alpha (Ann %)",
                color="Z_Score_Rank",
                text="index",
                labels={"index": "Ticker"},
                color_continuous_scale=["#38bdf8", "#1a1a22", ACCENT],
                color_continuous_midpoint=0
            )
            scatter_fig.update_traces(textposition='top center', marker=dict(size=11, line=dict(width=1, color='#0a0a0d')))
            style_plotly_dark(scatter_fig)
            st.plotly_chart(scatter_fig, use_container_width=True)

    with tab_heatmap:
        st.markdown("<div class='panel-title'>Factor Loadings Heatmap</div>", unsafe_allow_html=True)
        st.markdown("<span style='color:#9a9aa3;'>Systematic factor exposures across the universe — reveals size and value clustering at a glance.</span>", unsafe_allow_html=True)

        heatmap_fig = px.imshow(
            sorted_matrix[["Beta Market", "Beta Size (SMB)", "Beta Value (HML)"]],
            labels=dict(x="Factor", y="Ticker", color="Beta"),
            color_continuous_scale=["#38bdf8", "#14141a", ACCENT],
            color_continuous_midpoint=0,
            aspect="auto"
        )
        style_plotly_dark(heatmap_fig, height=520)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab_decomposition:
        col_ui, col_chart = st.columns([1, 2.2], gap="large")

        available_assets = sorted(sorted_matrix.index)

        with col_ui:
            selected_asset = st.selectbox("Select Equity", options=available_assets)
            asset_stats = sorted_matrix.loc[selected_asset]

            st.markdown(f"<div class='panel-title'>{selected_asset.replace('.NS', '')}</div>", unsafe_allow_html=True)

            st.metric("Annualized Alpha", f"{asset_stats['Alpha (Ann %)']:+.2f}%")
            st.metric("Market Beta", f"{asset_stats['Beta Market']:.3f}")

            st.markdown("#### Factor Tilts")
            st.write(f"**Size (SMB):** `{asset_stats['Beta Size (SMB)']:+.3f}`")
            st.write(f"**Value (HML):** `{asset_stats['Beta Value (HML)']:+.3f}`")
            st.write(f"**R²:** `{asset_stats['R-Squared']:.2%}`")
            st.write(f"**Sample:** `N={asset_stats['N (obs)']:.0f}`, `DoF={asset_stats['DoF']:.0f}`, HAC lags=`{asset_stats['HAC Lags']:.0f}`")

            is_significant = abs(asset_stats['t-stat Alpha']) > 1.96
            sig_label = "Statistically significant" if is_significant else "Not statistically significant"
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            st.markdown(
                f"Alpha t-stat = **`{asset_stats['t-stat Alpha']:.2f}`** "
                f"→ **{sig_label}** at the 95% level (|t| {'>' if is_significant else '≤'} 1.96)."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_chart:
            rolling_window = min(60, max(10, asset_stats['N (obs)'] // 3))
            if asset_stats['N (obs)'] < rolling_window + 10:
                st.warning(f"Insufficient history for rolling analysis ({asset_stats['N (obs)']:.0f} observations available).")
            else:
                with st.spinner("Computing rolling regressions..."):
                    rolling_df = calculate_rolling_exposures(returns[selected_asset], factors, window=int(rolling_window))

                    if rolling_df.empty:
                        st.info("Rolling exposures could not be computed for this asset and window.")
                    else:
                        roll_fig = px.line(
                            rolling_df.reset_index(),
                            x="Date",
                            y=["Market Factor", "Size Factor (SMB)", "Value Factor (HML)"],
                            labels={"value": "Beta", "variable": "Factor"},
                            color_discrete_map={
                                "Market Factor": "#38bdf8",
                                "Size Factor (SMB)": GREEN,
                                "Value Factor (HML)": ACCENT
                            }
                        )
                        style_plotly_dark(roll_fig)
                        roll_fig.update_layout(
                            title=f"{int(rolling_window)}-Day Rolling Betas: {selected_asset.replace('.NS', '')}",
                            hovermode="x unified",
                            legend=dict(orientation="h", yref="container", y=1.05, x=0.01)
                        )
                        st.plotly_chart(roll_fig, use_container_width=True)

    with tab_diagnostics:
        st.markdown("<div class='panel-title'>Diagnostics</div>", unsafe_allow_html=True)
        st.caption("All dropped tickers, data gaps, and run-level warnings are recorded here.")

        diag_df = diag.as_dataframe()
        if diag_df.empty:
            st.success("No tickers were dropped and no data gaps were detected in this run.")
        else:
            st.dataframe(diag_df, use_container_width=True, height=300)

        if diag.warnings:
            st.markdown("#### Warnings")
            for w in diag.warnings:
                st.write(f"- {w}")

        if diag.notes:
            st.markdown("#### Methodology Notes")
            for n in diag.notes:
                st.write(f"- {n}")

        st.markdown("#### Known Simplifications")
        st.markdown("""
- **Fundamentals are not point-in-time.** Shares outstanding and book value come from today's data, held static across the lookback window.
- **Risk-free rate is held constant** across the window rather than tracking actual historical policy-rate changes.
- **Universe is a current snapshot** — survivorship bias is present without a paid index-history source.
- **EWMA volatility (λ=0.94)** is a RiskMetrics approximation, not a fitted GARCH(1,1) model.
        """)

    total_possible_obs = len(raw_data) * len(stock_tickers)
    total_gap_obs = sum(diag.data_gaps.values())
    consistency_pct = 100 * (1 - total_gap_obs / total_possible_obs) if total_possible_obs else 100

    st.markdown(f"""
        <div class="footer-bar">
            <div>
                CACHE: <span class="ok">ACTIVE</span>
                <span class="sep">|</span>
                RISK-FREE: <span class="ok">{rbi_rate*100:.3f}%</span>
                <span class="sep">|</span>
                WINDOW: <span class="ok">{lookback_years} YEAR{'S' if lookback_years != 1 else ''}</span>
                <span class="sep">|</span>
                DATA COMPLETENESS: <span class="ok">{consistency_pct:.1f}%</span>
                <span class="sep">|</span>
                DROPPED: <span class="ok">{len(diag.dropped_tickers)}</span>
            </div>
            <div>
                NSE / Yahoo Finance via yfinance &nbsp;&nbsp;|&nbsp;&nbsp; {end_date}
            </div>
        </div>
    """, unsafe_allow_html=True)
