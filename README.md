# Quantitative Strategies Group (QSG) – Index Rebalancing Project

This repository contains the full solution and analysis for the Quantitative Trader Candidate Project assigned by the Quantitative Strategies Group (QSG). The objective is to develop and backtest trading strategies that exploit opportunities arising from predictable index rebalancing events (e.g., S&P 400/500/600 additions).

##  Project Overview

Throughout the year, major indexes add, remove, and update holdings. These index rebalancing events are often anticipated, creating temporary liquidity and price inefficiencies that traders can exploit. This project analyzes a dataset of index events since May 2022 and explores several systematic trading strategies around these events.

##  Repository Structure
<pre> ``` 
QSG-Index-Rebalance-Project/
│
├── data/           # Raw and cleaned data files (e.g., Excel event list)
├── src/            # All Python scripts for backtesting, data prep, analysis
├── results/        # Output files (performance stats, tables, figures)
│   └── figures/    # Generated plots and figures
├── report/         # Final write-up/report (Notebook and PDF)
├── requirements.txt# Python dependencies
└── README.md       # This file
``` </pre>

## Strategies Implemented

- **Post-Announcement Momentum:**  
  Buy stocks added to the index after announcement and exit before the official trade date.
- **Event Day Reversion:**  
  Trade stocks on the index event date, betting on mean-reversion vs. the index return.
- **Buy-and-Hold:**  
  Hold from announcement (first tradable day) through the trade date.
- **Holding Period Sweep:**  
  Test different holding periods to optimize performance.
- **Hedging and Carry:**  
  Apply hedges (SPY) and include overnight costs per QSG instructions.

## Output & Results

- **Performance metrics:** Net P&L, average return, win rate, Sharpe ratio, max drawdown, etc.
- **Plots:** Cumulative P&L, drawdowns, and other visualizations (see `results/figures/`)
- **Final Report:** See `report/` for full write-up and discussion.

## Notes & Assumptions

- End-of-day price data sourced via [yfinance](https://github.com/ranaroussi/yfinance).
- Transaction and carry costs modeled per QSG guidelines.
- Portfolio and risk limits (e.g., max $250k/trade, 1% ADV) enforced.
- Only surviving tickers included (delisted stocks removed).
- Provided datasets are not redistributed here for data privacy.

## How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Minouneshan/QSG-Index-Rebalance-Project.git
    cd QSG-Index-Rebalance-Project
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the dataset:**
    - Place the provided Excel index event file in the `data/` folder.

4. **Run the main backtest script:**
    ```bash
    python src/backtest_strategies.py
    ```

## Modular Code Structure & Approach

| Module                      | Purpose                                                      |
|-----------------------------|--------------------------------------------------------------|
| `src/strategies.py`         | Core trading logic: all event trading strategies implemented |
| `src/sweeps.py`             | Parameter sweeps for strategies                             |
| `src/optimizer.py`          | ML/grid optimization routines                               |
| `src/backtest_strategies.py`| Main pipeline: event filtering, running, saving results      |
| `src/utils.py`, `data_prep.py` | Utilities and data loading/cleaning                      |

The codebase is fully modular and robust, with each component responsible for a specific part of the pipeline:

- `src/strategies.py`: Implements all core trading strategies as functions. Each returns both detailed P&L DataFrames and summary statistics, with robust handling for missing/invalid data.
- `src/sweeps.py`: Parameter sweep logic for strategies, ensuring all parameter keys are stringified for JSON compatibility. Includes wrappers for scalar parameter sweeps.
- `src/optimizer.py`: Grid and ML-based optimization routines, with error handling for empty/missing results and wrappers for parameter adaptation.
- `src/backtest_strategies.py`: Main pipeline for event filtering, parameter casting, running strategies, and saving results. Skips/prints messages for empty or invalid results.
- `src/utils.py` & `src/data_prep.py`: Utility functions for data loading, cleaning, and common operations.

### Workflow
1. **Data Preparation:** Load and clean event and price data using `data_prep.py` and `utils.py`.
2. **Strategy Backtesting:** Run strategies via `backtest_strategies.py`, which calls modular strategy functions implemented in `strategies.py` and handles event filtering and parameter casting.
3. **Parameter Sweeps & Optimization:** Use `sweeps.py` and `optimizer.py` for systematic parameter exploration and ML-based optimization, all of which ultimately invoke strategy logic from `strategies.py`.
4. **Results & Reporting:** All results (P&L, summaries, plots) are saved to `results/` and visualized in the Jupyter notebook (`report/QSG Index Rebalance Analysis.ipynb`).
5. **Notebook:** The notebook summarizes methodology, results, and includes all main equity curve plots, with clear markdown and code cells for reproducibility.

### Robustness & Reproducibility
- All functions handle missing data and empty results gracefully.
- Parameter types are validated and cast as needed.
- All outputs are reproducible and figures are saved for reporting.

---

For further details, see the code comments and the final notebook in `report/`.
