# 📈 Factor Analytics Terminal

An interactive quantitative finance dashboard that estimates Fama-French factor exposures for Indian equities using historical market data.

Built with **Python**, **Streamlit**, **Plotly**, and **Statsmodels**, the application performs cross-sectional factor analysis, rolling regressions, and equity screening through an institutional-style interface.

---

## Links

- **Live App:** https://factor-analytics-terminal-nse50.streamlit.app/
- **GitHub Repository:** https://github.com/Ausluxx/factor-analytics-terminal

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

<img width="1847" height="646" alt="image" src="https://github.com/user-attachments/assets/f6bb154c-61e5-4137-b783-9fae0f9ab63b" />

<img width="1871" height="777" alt="image" src="https://github.com/user-attachments/assets/fe7aa67c-50ca-467a-afb6-1d602cfa6baa" />

<img width="1800" height="802" alt="image" src="https://github.com/user-attachments/assets/d0f38ccb-a60d-4caa-bbce-31d62c043201" />

<img width="1502" height="807" alt="image" src="https://github.com/user-attachments/assets/98b80ed9-1aa4-4659-bef2-e32b45882ca2" />

<img width="1840" height="727" alt="image" src="https://github.com/user-attachments/assets/60d760e1-0d1d-4332-a6ec-ea38786f0e7d" />

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
