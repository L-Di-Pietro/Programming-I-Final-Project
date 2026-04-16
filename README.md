# Black-Scholes-Merton Option Pricing & Implied Volatility Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Finance-Quantitative-0A66C2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

> **A comprehensive Python-based quantitative finance framework** implementing the Black-Scholes-Merton (BSM) model from first principles, featuring real-time option chain data acquisition, implied volatility extraction via Newton-Raphson, volatility smile/skew/term-structure analysis, and rigorous model validation — applied to SPY and seven major technology stocks.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Authors](#2-authors)
3. [Mathematical Background](#3-mathematical-background)
   - [The Black-Scholes-Merton Model](#31-the-black-scholes-merton-model)
   - [Vega — The Volatility Greek](#32-vega--the-volatility-greek)
   - [Implied Volatility & Newton-Raphson](#33-implied-volatility--newton-raphson)
   - [The Volatility Surface](#34-the-volatility-surface)
4. [Project Structure](#4-project-structure)
5. [Notebook Walkthrough](#5-notebook-walkthrough)
6. [Key Functions](#6-key-functions)
7. [Data & Assets](#7-data--assets)
8. [Empirical Findings](#8-empirical-findings)
9. [Installation & Requirements](#9-installation--requirements)
10. [Usage Guide](#10-usage-guide)
11. [Important Notes on Running the Notebook](#11-important-notes-on-running-the-notebook)
12. [Validation & Accuracy](#12-validation--accuracy)
13. [Limitations & Future Extensions](#13-limitations--future-extensions)
14. [Technical Skills Demonstrated](#14-technical-skills-demonstrated)
15. [References](#15-references)

---

## 1. Project Overview

This project builds a **complete, production-quality quantitative finance pipeline** entirely in Python. Starting from the theoretical derivation of the Black-Scholes-Merton PDE, it implements:

- A from-scratch BSM pricing engine for European calls and puts
- An analytical Vega calculator used as the derivative in the numerical solver
- A Newton-Raphson implied volatility (IV) extractor capable of processing thousands of option contracts
- A full data-acquisition layer pulling live option chains from **Yahoo Finance** (`yfinance`) for **SPY** (S&P 500 ETF) and **seven major technology stocks**: `TSLA`, `NVDA`, `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `META`
- Publication-quality visualisations of the **volatility smile**, **volatility skew**, and **volatility term structure**
- A rigorous validation suite: round-trip repricing tests and cross-validation against Yahoo Finance–reported implied volatilities, benchmarked with RMSE, MAE, MAPE, and Bland-Altman agreement analysis

The result is a self-contained research notebook that goes from raw market data to market-microstructure insights, documenting every modelling choice along the way.

---

## 2. Authors

| Name | Role |
|---|---|
| **Blum** | Co-author |
| **DiNoia** | Co-author |
| **Di Pietro, L.** | Co-author & repository owner |

> Final project for **Programming I** course.

---

## 3. Mathematical Background

### 3.1 The Black-Scholes-Merton Model

The BSM model (Black & Scholes 1973; Merton 1973) provides the first closed-form solution for European option pricing. Under the assumption that the underlying asset follows a **Geometric Brownian Motion**:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

and invoking risk-neutral valuation, the prices of European calls and puts are:

$$C(S, K, T, r, \sigma) = S \cdot N(d_1) - K e^{-rT} N(d_2)$$

$$P(S, K, T, r, \sigma) = K e^{-rT} N(-d_2) - S \cdot N(-d_1)$$

where:

$$d_1 = \frac{\ln(S/K) + \bigl(r + \tfrac{1}{2}\sigma^2\bigr)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

| Symbol | Description |
|--------|-------------|
| $S$ | Current spot price of the underlying asset |
| $K$ | Strike price of the option |
| $T$ | Time to expiration (in years) |
| $r$ | Continuously compounded risk-free rate |
| $\sigma$ | Annualised volatility of the underlying |
| $N(\cdot)$ | Standard normal CDF |

**Key model assumptions:**
1. Asset price follows Geometric Brownian Motion with constant drift and volatility
2. No dividends during the option's lifetime
3. Frictionless, complete markets with continuous trading
4. Constant risk-free rate and volatility over the option's life
5. European-style exercise only (at expiration)
6. Log-normally distributed asset prices

**Put-Call Parity** is preserved: $C - P = S - Ke^{-rT}$

---

### 3.2 Vega — The Volatility Greek

Vega is the partial derivative of the option price with respect to volatility:

$$\mathcal{V} = \frac{\partial C}{\partial \sigma} = \frac{\partial P}{\partial \sigma} = S \sqrt{T} \cdot \phi(d_1)$$

where $\phi(\cdot)$ is the standard normal PDF. Vega is identical for calls and puts with the same parameters and is maximised at-the-money. It serves two critical roles in this project:

1. **Newton-Raphson denominator**: provides the analytical gradient for the IV solver, enabling quadratic convergence
2. **Risk sensitivity**: quantifies how much an option's value changes per 1% move in volatility

---

### 3.3 Implied Volatility & Newton-Raphson

Because the BSM formula cannot be algebraically inverted to solve for $\sigma$, implied volatility (IV) is defined as the root of:

$$f(\sigma) = C_{\text{BS}}(S, K, T, r, \sigma) - C_{\text{market}} = 0$$

This project solves this root-finding problem using the **Newton-Raphson algorithm**:

$$\sigma_{n+1} = \sigma_n - \frac{C_{\text{BS}}(\sigma_n) - C_{\text{market}}}{\mathcal{V}(\sigma_n)}$$

**Algorithm properties:**
- **Quadratic convergence** — number of correct digits doubles per iteration
- Typically converges in **5–10 iterations** to machine precision ($< 10^{-8}$ error)
- Uses $\sigma_0 = 0.20$ (20%) as the default initial guess
- Bounded between $0^+$ and $5.0$ (500%) to prevent numerical overflow
- Returns `None` on failure (expired option, zero Vega, arbitrage violation, non-positive price)

---

### 3.4 The Volatility Surface

The BSM model predicts that all options on the same underlying should share a **constant implied volatility**. Empirically, this is violated: IV varies systematically across strikes and maturities, forming the **volatility surface** $\sigma_{\text{IV}}(K, T)$.

**Volatility Smile / Skew** — IV plotted vs. strike for a fixed expiration:
- Equity / index options exhibit a pronounced **reverse skew**: deep OTM puts carry materially higher IV than ATM or OTM calls
- Intensified after the 1987 crash ("Black Monday") when portfolio insurance demand for downside protection became structural

**Volatility Term Structure** — ATM IV vs. time to expiration:
- **Upward sloping**: long-dated options priced with higher uncertainty
- **Downward sloping**: near-term events (earnings, FOMC) elevate short-dated IV
- **Flat**: markets expect consistent volatility across horizons

---

## 4. Project Structure

```
Programming-I-Final-Project/
│
├── BSM_Programming_Project_Blum_DiNoia_DiPietro.ipynb   # Main notebook (all code & analysis)
│
├── Data_Option_Chain_Modified_Stocks/      # Raw option chain CSVs (per ticker, per expiry)
├── Data_IV_Calculated_Stocks/              # Processed IV data CSVs (per ticker)
│
├── Graphs_Volatility_Smile_Skew/           # Volatility smile/skew plots (per ticker)
├── Graphs_Volatility_Term_Structure/       # Term structure plots (per ticker)
├── Graphs_IV_Validation_Repricing/         # Repricing error / round-trip test plots
└── Graphs_IV_Validation_CrossValidation/   # Cross-validation vs. YFinance IV plots
```

> **Note:** The six output directories (`Data_*` and `Graphs_*`) are created automatically by the notebook at runtime. They are only populated when the **Master Control Switch** `SAVE_FILES = True` is set in Cell 1.

---

## 5. Notebook Walkthrough

The notebook is organised into five major parts, each preceded by detailed markdown cells explaining the theory before the code:

### Cell 1 — Master Control Switch & Preliminary Notes
Sets the global `SAVE_FILES` flag. When `False` (default), all file I/O operations (`os.makedirs`, `pd.DataFrame.to_csv`, `plt.savefig`) are silently replaced with no-ops, allowing a clean read-only run. When `True`, all output directories and files are created on disk.

### Part 1 — Mathematical Foundations & BSM Implementation
- Full derivation of the Black-Scholes PDE and closed-form solution
- Economic intuition behind $N(d_1)$, $N(d_2)$, and Put-Call Parity
- **`black_scholes()`** — European call/put pricing function
- **`vega()`** — Vega calculation used as the Newton-Raphson gradient

### Part 2 — Implied Volatility Extraction via Newton-Raphson
- Framing IV extraction as a root-finding problem
- Convergence theory and comparison with alternative methods (bisection, Brent, secant)
- **`implied_volatility()`** — Full Newton-Raphson solver with input validation, iteration controls, and boundary enforcement
- Round-trip validation test (error < $10^{-8}$)

### Part 3 — Data Acquisition & Processing
- YFinance API integration for live option chain retrieval
- Risk-free rate sourcing (U.S. Treasury yields)
- Mid-price calculation: $\text{midPrice} = (\text{bid} + \text{ask}) / 2$, with fallback to `lastPrice` when markets are closed
- Data cleaning pipeline: filtering illiquid contracts, zero-volume options, and arbitrage violations
- Optional Filters 3 & 4 in Cell 14 for tighter data quality control
- Batch IV computation across all tickers, strikes, and expiration dates

### Part 4 — Empirical Analysis & Visualisation
- **Volatility smile/skew** charts per ticker and expiration
- **Volatility term structure** charts (ATM IV vs. days to expiry)
- **Cross-asset comparison**: SPY index vs. single-stock technology sector IVs
- All charts follow a consistent, publication-quality style (`seaborn` whitegrid, `matplotlib`)

### Part 5 — Model Validation & Key Findings
- **Repricing analysis**: BSM prices recomputed from extracted IVs vs. original market prices
- **Cross-validation**: Our computed IVs vs. YFinance-reported IVs
- Statistical error metrics: RMSE, MAE, MAPE
- Bland-Altman agreement plots
- **Cell 35 — Key Empirical Findings**: Extended discussion of observed market phenomena, model limitations, and broader implications

---

## 6. Key Functions

### `black_scholes(S, K, T, r, sigma, option_type='call')`

Computes the Black-Scholes-Merton theoretical price for a European option.

| Parameter | Type | Description |
|-----------|------|-------------|
| `S` | `float` | Current spot price |
| `K` | `float` | Strike price |
| `T` | `float` | Time to expiration in years |
| `r` | `float` | Risk-free rate (decimal, e.g. `0.05` = 5%) |
| `sigma` | `float` | Annualised volatility (decimal) |
| `option_type` | `str` | `'call'` or `'put'` |

**Returns:** `float` — theoretical option price  
**Edge case:** Returns intrinsic value directly when `T ≤ 0`

```python
# Example
price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.20, option_type='call')
# → $10.45
```

---

### `vega(S, K, T, r, sigma)`

Computes the option's Vega (sensitivity to volatility), identical for calls and puts.

**Returns:** `float` — Vega value  
**Returns `0`** when `T ≤ 0`

```python
v = vega(S=100, K=100, T=1, r=0.05, sigma=0.20)
# → 37.52  (a 1% vol increase ≈ +$0.375 in option value)
```

---

### `implied_volatility(market_price, S, K, T, r, option_type, initial_sigma, tolerance, max_iterations)`

Extracts implied volatility from an observed market price using Newton-Raphson iteration.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `market_price` | — | Observed market price of the option |
| `S` | — | Current spot price |
| `K` | — | Strike price |
| `T` | — | Time to expiration in years |
| `r` | — | Risk-free rate |
| `option_type` | `'call'` | `'call'` or `'put'` |
| `initial_sigma` | `0.20` | Starting guess for volatility |
| `tolerance` | `1e-6` | Convergence criterion |
| `max_iterations` | `100` | Maximum iterations before failure |

**Returns:** `float` — implied volatility (annualised decimal), or `None` if convergence fails

```python
iv = implied_volatility(market_price=12.34, S=100, K=100, T=1, r=0.05, option_type='call')
# → 0.2500  (25.00%)
```

**Failure conditions (returns `None`):**
- `market_price ≤ 0`
- `T ≤ 0` (expired)
- `market_price < intrinsic_value` (no-arbitrage violation)
- Vega falls below `1e-10` (numerical instability)
- Does not converge within `max_iterations`

---

## 7. Data & Assets

### Tickers Covered

| Ticker | Description | Type |
|--------|-------------|------|
| `SPY` | SPDR S&P 500 ETF Trust | Index ETF |
| `TSLA` | Tesla, Inc. | Single-stock |
| `NVDA` | NVIDIA Corporation | Single-stock |
| `AAPL` | Apple Inc. | Single-stock |
| `MSFT` | Microsoft Corporation | Single-stock |
| `AMZN` | Amazon.com, Inc. | Single-stock |
| `GOOGL` | Alphabet Inc. (Class A) | Single-stock |
| `META` | Meta Platforms, Inc. | Single-stock |

### Data Source

All option chain data is fetched live via the **[yfinance](https://github.com/ranaroussi/yfinance)** Python library, which queries Yahoo Finance. The risk-free rate is proxied by the current U.S. Treasury yield.

### Reference Outputs

The pre-saved outputs in the repository were generated on **26 November 2025 at 21:48 CET** during live US market hours, with Filters 3 & 4 in Cell 14 **not applied** (to maximise the dataset size for demonstration purposes).

### Output Directories (created at runtime)

| Directory | Contents |
|-----------|----------|
| `Data_Option_Chain_Modified_Stocks/` | Cleaned option chain CSV files (one per ticker/expiry combination) |
| `Data_IV_Calculated_Stocks/` | Processed DataFrames with computed IV columns |
| `Graphs_Volatility_Smile_Skew/` | Volatility smile & skew plots |
| `Graphs_Volatility_Term_Structure/` | Term structure plots |
| `Graphs_IV_Validation_Repricing/` | Round-trip repricing validation charts |
| `Graphs_IV_Validation_CrossValidation/` | Cross-validation vs. YFinance IV charts |

---

## 8. Empirical Findings

The following key market phenomena were observed and documented in Cell 35 of the notebook (based on data from 26 November 2025):

### Volatility Skew
A pronounced **reverse skew** is consistently observed across all tickers: out-of-the-money (OTM) puts carry materially higher implied volatility than at-the-money options, which in turn trade above OTM calls. This pattern reflects institutional demand for downside protection ("crash insurance"), risk premiums for fat-tail events, and the structural supply-demand imbalance in the options market since the 1987 crash.

### Cross-Asset IV Hierarchy
Single-stock implied volatilities are uniformly higher than index (SPY) IVs, reflecting diversification: idiosyncratic risks that inflate single-stock IV are partially cancelled in the index. Among individual stocks, high-beta, high-growth names (e.g., `TSLA`, `NVDA`) exhibit meaningfully higher IVs than more stable large-caps (`AAPL`, `MSFT`).

### Volatility Term Structure
ATM implied volatility varies systematically with time to expiration — shorter-dated options often embed higher IV when binary events (earnings, macro data releases) are imminent, while longer-dated options reflect smoother long-run uncertainty.

### Model Accuracy
The Newton-Raphson IV solver achieves:
- **Round-trip error** < $10^{-8}$ in controlled tests
- **Low RMSE and MAE** versus Yahoo Finance–reported IVs under liquid, open-market conditions
- Predictable degradation when markets are closed (bid/ask = 0, YFinance IVs = 0 → cross-validation becomes uninformative)

---

## 9. Installation & Requirements

### Python Version
Python **3.10** (as used during development; 3.9+ should be compatible)

### Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn yfinance pytz
```

| Library | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.23 | Numerical computations, array operations |
| `scipy` | ≥ 1.9 | `norm.cdf`, `norm.pdf` for BSM formula |
| `pandas` | ≥ 1.5 | Data manipulation and option chain processing |
| `matplotlib` | ≥ 3.6 | Plotting and figure export |
| `seaborn` | ≥ 0.12 | Enhanced statistical visualisations |
| `yfinance` | ≥ 0.2 | Yahoo Finance option chain & price data |
| `pytz` | ≥ 2022 | CET timezone handling for file timestamps |

### Optional: Virtual Environment (Recommended)

```bash
# Create environment
python -m venv bsm-env

# Activate (Linux/macOS)
source bsm-env/bin/activate

# Activate (Windows)
bsm-env\Scripts\activate

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn yfinance pytz
```

---

## 10. Usage Guide

### 1. Clone the Repository

```bash
git clone https://github.com/L-Di-Pietro/Programming-I-Final-Project.git
cd Programming-I-Final-Project
```

### 2. Launch Jupyter

```bash
jupyter notebook BSM_Programming_Project_Blum_DiNoia_DiPietro.ipynb
```

### 3. Configure the Master Control Switch (Cell 1)

```python
SAVE_FILES = False   # Default: run without saving any files to disk
SAVE_FILES = True    # Set to True to export all CSVs and plots to output folders
```

### 4. Run All Cells

Use **Kernel → Restart & Run All** for a clean end-to-end execution.

> **Tip:** For best results, run the notebook **during US equity market hours (15:30–22:00 CET; ideally 16:15–22:00 CET)**. See the [next section](#11-important-notes-on-running-the-notebook) for details.

### 5. Using Individual Functions

The three core functions can be imported and used standalone once the notebook has been executed (or copy-pasted into your own scripts):

```python
# Price a European call
price = black_scholes(S=580, K=580, T=30/365, r=0.053, sigma=0.18, option_type='call')

# Calculate Vega
v = vega(S=580, K=580, T=30/365, r=0.053, sigma=0.18)

# Extract implied volatility from a market price
iv = implied_volatility(market_price=price, S=580, K=580, T=30/365, r=0.053, option_type='call')
```

---

## 11. Important Notes on Running the Notebook

> ⚠️ **Please read before running.**

### Market Hours Requirement

It is **strongly recommended** to run the notebook while **US equity markets are open** (approximately **15:30–22:00 CET**, with the most liquid period from **16:15–22:00 CET**).

**Why this matters:**
- The notebook fetches **real-time** bid/ask prices and implied volatilities from Yahoo Finance
- When markets are **closed**, bid and ask prices are `0`, causing the midPrice calculation to be inaccurate
- The notebook falls back to `lastPrice` when bid/ask are unavailable, but this is only an approximation
- Yahoo Finance reports `impliedVolatility = 0.0` for all options outside market hours, making the cross-validation in the final cells **entirely uninformative**

### Outputs are Time-Stamped

All saved files are automatically timestamped using CET time (format: `DD_MM_YYYY_HHMM`), ensuring outputs from different runs do not overwrite each other.

### Pre-saved Output Context

The pre-generated outputs already committed to the repository were produced on **26.11.2025 at 21:48 CET** (during live market hours). Filters 3 and 4 in Cell 14 were **not applied** for those runs, yielding a larger dataset for demonstration purposes. These filters remain off by default but can be toggled independently.

---

## 12. Validation & Accuracy

Two independent validation approaches are implemented:

### Round-Trip Repricing Test
1. Price an option with a known true volatility $\sigma_{\text{true}}$ using `black_scholes()`
2. Feed the price into `implied_volatility()` to recover $\hat{\sigma}$
3. Check: $|\hat{\sigma} - \sigma_{\text{true}}| < \epsilon$

**Result:** Error consistently below $10^{-8}$ across all tested parameter combinations, confirming that both `black_scholes()` and `implied_volatility()` are correctly implemented.

### Cross-Validation vs. Yahoo Finance IVs
- For each option contract where YFinance provides a non-zero IV, our computed IV is compared
- **Error metrics computed:** RMSE, MAE, MAPE
- **Bland-Altman plots** assess limits of agreement and systematic bias
- Results are meaningful only during market hours (YFinance IVs are zero when markets are closed)

---

## 13. Limitations & Future Extensions

### Current Limitations

| Limitation | Description |
|------------|-------------|
| **Constant volatility** | BSM assumes $\sigma$ is constant — violated by the volatility smile itself |
| **No dividends** | Discrete dividend payments are not modelled (relevant for high-yield stocks) |
| **European exercise only** | American options with early exercise premiums are not supported |
| **Log-normal returns** | Fat tails and jump risk are not captured; deep OTM options may produce IV failures |
| **Market hours dependency** | Data quality degrades significantly outside US trading hours |
| **Static risk-free rate** | A single flat rate is used; no term structure of interest rates |

### Suggested Future Extensions

The project explicitly identifies the following as natural next steps:

- **Stochastic Volatility Models**: Heston (1993), SABR (Hagan et al. 2002)
- **Local Volatility**: Dupire (1994), Derman-Kani (1994)
- **Jump-Diffusion Models**: Merton (1976), Bates (1996), Kou (2002)
- **Rough Volatility**: Fractional Brownian motion models (Gatheral et al. 2018)
- **American Options**: Binomial trees, Least-Squares Monte Carlo (Longstaff-Schwartz)
- **Dividend Adjustment**: Continuous dividend yield or discrete dividend schedule
- **Machine Learning**: IV forecasting, anomaly detection, volatility surface interpolation
- **High-Frequency Analysis**: Intraday volatility patterns and microstructure effects
- **Cross-Asset Relationships**: Equity-FX, equity-credit volatility correlations

---

## 14. Technical Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Mathematical Finance** | Derivatives pricing theory, stochastic calculus foundations, risk-neutral valuation, Greeks |
| **Numerical Methods** | Newton-Raphson root-finding, convergence analysis, numerical stability, edge-case handling |
| **Python Programming** | Clean functional design, vectorisation, API integration (`yfinance`), error handling, monkey-patching |
| **Data Science** | Large-scale option chain processing, statistical analysis, outlier detection, quality control |
| **Visualisation** | Publication-quality plots (`matplotlib`, `seaborn`), multi-panel layouts, interpretable financial charts |
| **Validation & Testing** | Round-trip testing, cross-validation, statistical benchmarking (RMSE/MAE/MAPE), Bland-Altman analysis |
| **Financial Markets** | Options mechanics, market microstructure, institutional hedging behaviour, derivatives trading |

---

## 15. References

- Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637–654.
- Merton, R. C. (1973). *Theory of Rational Option Pricing*. Bell Journal of Economics and Management Science, 4(1), 141–183.
- Heston, S. L. (1993). *A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options*. Review of Financial Studies, 6(2), 327–343.
- Dupire, B. (1994). *Pricing with a Smile*. Risk, 7(1), 18–20.
- Merton, R. C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous*. Journal of Financial Economics, 3(1–2), 125–144.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). *Volatility is Rough*. Quantitative Finance, 18(6), 933–949.
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley.

---

<p align="center">
  <i>Built with Python · Powered by Yahoo Finance · Grounded in theory</i>
</p>
