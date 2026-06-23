# 📈 Factor Analytics Terminal

An interactive quantitative finance dashboard that estimates Fama-French factor exposures for Indian equities using historical market data.

Built with **Python**, **Streamlit**, **Plotly**, and **Statsmodels**, the application performs cross-sectional factor analysis, rolling regressions, and equity screening through an institutional-style interface.

---

## Links

- 🚀 **Live App:** https://factor-analytics-terminal.streamlit.app
- 💻 **GitHub Repository:** https://github.com/Ausluxx/factor-analytics-terminal

---

## Features

* Fama-French Three-Factor Model
* Alpha and Beta estimation using OLS regression
* Newey-West (HAC) robust standard errors
* Rolling factor exposure analysis
* EWMA and historical volatility estimation
* Interactive equity screener
* Factor exposure heatmap
* Diagnostics for missing data and model assumptions
* Support for the Nifty 50 universe or custom tickers

---

## Technologies

* Python
* Streamlit
* Pandas
* NumPy
* Statsmodels
* Plotly
* yFinance

---

## Methodology

The application:

1. Downloads historical adjusted prices from Yahoo Finance.
2. Retrieves company fundamentals required for factor construction.
3. Builds the Fama-French Market, SMB, and HML factors.
4. Estimates stock-specific factor exposures using Ordinary Least Squares.
5. Applies Newey-West HAC standard errors.
6. Computes annualized alpha, factor betas, volatility, and R².
7. Visualizes rolling factor exposures and cross-sectional rankings.

---

## Dashboard

The application includes:

* Equity Screener
* Risk vs Return Visualization
* Factor Loading Heatmap
* Rolling Beta Analysis
* Diagnostics & Data Quality Report

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/factor-analytics-terminal.git
cd factor-analytics-terminal
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run factor_terminal.py
```

---

## Project Structure

```
factor-analytics-terminal/
│
├── factor_terminal.py      # Streamlit interface
├── factor_engine.py        # Factor calculations and regressions
├── requirements.txt
└── README.md
```

---

## Data Source

* Yahoo Finance (via yfinance)

---

## Limitations

* Uses current Nifty 50 constituents (survivorship bias).
* Fundamentals are treated as static over the analysis window.
* Risk-free rate is user-specified and held constant.
* Intended for educational and research purposes.

---

## Future Improvements

* Five-Factor Fama-French Model
* Carhart Momentum Factor
* Portfolio Optimization
* Risk Attribution
* Performance Backtesting
* PDF Research Report Export
* Multi-Country Equity Universes

---

## License

MIT License
