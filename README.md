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
├── results/        # Output files (performance stats, tables)
│   └── figures/    # Generated plots and figures
├── report/         # Final write-up/report (Markdown or PDF)
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
    git clone https://github.com/yourusername/QSG-Index-Rebalance-Project.git
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
