# src/strategies.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import sharpe_ratio, max_drawdown, sortino_ratio, win_rate, save_summary

os.makedirs("results/figures", exist_ok=True)

# ===== 1. Post-Announcement Momentum =====
def post_announcement_momentum(
    events, price, 
    hold_days=None,        # <-- None means default behavior, else # of days after entry
    max_usd=250_000, 
    tran_cost=0.01, 
    save_fig=True, 
    show_fig=False
):
    """
    Implements post-announcement momentum strategy.
    Satisfies assignment requirements:
        - Liquidity constraint (<=1% ADV)
        - Transaction cost ($0.01/share)
        - Entry: Open after announcement; Exit: Day before event
        - Risk: Returns, Sharpe, Drawdown, etc. computed
        - All exclusions tracked
    Returns: DataFrame, summary metrics, exclusions count
    """
    results = []
    exclusions = {'no_price': 0, 'illiquid': 0, 'too_small': 0}
    for idx, row in events.iterrows():
        ticker = row['Yahoo_Ticker']
        entry_date = row['first_tradable']
        if hold_days is not None:
            exit_date = entry_date + pd.Timedelta(days=hold_days-1)
            # Make sure we never go past the Trade Date minus one day!
            if exit_date > row['Trade Date'] - pd.Timedelta(days=1):
                exit_date = row['Trade Date'] - pd.Timedelta(days=1)
        else:
            exit_date = row['Trade Date'] - pd.Timedelta(days=1)
        try:
            entry = price.loc[(entry_date, ticker)]
            exit_ = price.loc[(exit_date, ticker)]
        except KeyError:
            exclusions['no_price'] += 1
            continue
        adv = entry.get('ADV20', np.nan)
        if np.isnan(adv) or adv <= 0:
            exclusions['illiquid'] += 1
            continue
        px_open = entry['Open']
        px_close = exit_['Close']
        size_shares = min(int(max_usd / px_open), int(0.01 * adv))
        if size_shares < 1:
            exclusions['too_small'] += 1
            continue
        gross_pnl = (px_close - px_open) * size_shares
        total_cost = tran_cost * size_shares * 2
        net_pnl = gross_pnl - total_cost
        ret = net_pnl / (px_open * size_shares)
        results.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_open': px_open,
            'exit_close': px_close,
            'shares': size_shares,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return': ret,
            'usd_invested': px_open * size_shares,
            'trade_days': (exit_date - entry_date).days + 1,
        })
    bt = pd.DataFrame(results)
    print("\n=== Post-Announcement Momentum Results ===")
    if bt.empty:
        print("No trades generated.")
        return bt, {}
    print(bt.describe())
    print(bt.head(10))
    total_pnl = bt['net_pnl'].sum()
    avg_return = bt['return'].mean()
    win_rate = (bt['net_pnl'] > 0).mean()
    avg_holding = bt['trade_days'].mean()
    std_return = bt['return'].std()
    sharpe = sharpe_ratio(bt['return'])
    sortino = sortino_ratio(bt['return'])
    mdd = max_drawdown(bt['net_pnl'].cumsum())
    num_trades = len(bt)
    
    summary = {
        "total_net_pnl": total_pnl,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "avg_holding_days": avg_holding,
        "std_return": std_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "num_trades": num_trades,
        "exclusions": exclusions,
    }
    # PRINT summary for logs
    print(f"SUMMARY: {summary}")
    save_summary(summary, "post_announcement_momentum")

    bt['exit_date'] = pd.to_datetime(bt['exit_date'])
    bt_grouped = bt.groupby('exit_date')['net_pnl'].sum().sort_index().cumsum()
    bt_grouped.plot(
        title='Cumulative Net P&L (Post-Announcement Momentum)', 
        ylabel='USD', xlabel='Date', grid=True, legend=False, figsize=(10,4)
    )
    plt.tight_layout()
    if save_fig:
        plt.savefig("results/figures/post_announcement_momentum_pnl.png")
    if show_fig:
        plt.show()
    plt.close()
    return bt, summary 

# ===== 2. Event Day Reversion =====
def event_day_reversion(
    events, price, 
    threshold=0.001, 
    hold_days=None,             # NEW PARAM
    max_usd=250_000, 
    tran_cost=0.01, 
    save_fig=True, 
    show_fig=False
):
    reversion_results = []
    exclusions = {'no_price': 0, 'illiquid': 0, 'too_small': 0, 'missing_spy': 0}
    for idx, row in events.iterrows():
        ticker = row['Yahoo_Ticker']
        event_date = row['Trade Date']
        entry_date = event_date
        # If holding longer than event day, update exit date
        if hold_days is not None:
            exit_date = entry_date + pd.Timedelta(days=hold_days-1)
        else:
            exit_date = entry_date

        try:
            stk_entry = price.loc[(entry_date, ticker)]
            stk_exit = price.loc[(exit_date, ticker)]
            adv = stk_entry.get('ADV20', np.nan)
            stk_open = stk_entry['Open']
            stk_close = stk_exit['Close']
            spy_open = row.get('spy_open', np.nan)
            spy_close = row.get('spy_close', np.nan)
            if pd.isna(spy_open) or pd.isna(spy_close):
                exclusions['missing_spy'] += 1
                continue
        except KeyError:
            exclusions['no_price'] += 1
            continue

        if np.isnan(adv) or adv <= 0:
            exclusions['illiquid'] += 1
            continue

        stk_ret = (stk_close - stk_open) / stk_open
        spy_ret = (spy_close - spy_open) / spy_open
        perf_delta = stk_ret - spy_ret

        if perf_delta > threshold:
            direction = -1
        elif perf_delta < -threshold:
            direction = 1
        else:
            continue

        size_shares = min(int(max_usd / stk_open), int(0.01 * adv))
        if size_shares < 1:
            exclusions['too_small'] += 1
            continue

        gross_pnl = direction * (stk_close - stk_open) * size_shares
        total_cost = tran_cost * size_shares * 2
        net_pnl = gross_pnl - total_cost
        ret = net_pnl / (stk_open * size_shares)
        reversion_results.append({
            'ticker': ticker,
            'event_date': entry_date,
            'exit_date': exit_date,
            'direction': direction,
            'stk_open': stk_open,
            'stk_close': stk_close,
            'spy_open': spy_open,
            'spy_close': spy_close,
            'perf_delta': perf_delta,
            'shares': size_shares,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return': ret,
            'usd_invested': stk_open * size_shares,
            'trade_days': (exit_date - entry_date).days + 1,
        })
    rv = pd.DataFrame(reversion_results)
    if rv.empty:
        print("No trades generated.")
        return rv, {}
    total_pnl = rv['net_pnl'].sum()
    avg_return = rv['return'].mean()
    win = win_rate(rv['net_pnl'])
    sharpe = sharpe_ratio(rv['return'])
    sortino = sortino_ratio(rv['return'])
    mdd = max_drawdown(rv['net_pnl'].cumsum())
    summary = {
        "total_net_pnl": total_pnl,
        "avg_return": avg_return,
        "win_rate": win,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "num_trades": len(rv),
        "exclusions": exclusions,
    }
    print(f"SUMMARY: {summary}")
    save_summary(summary, "event_day_reversion")
    if not rv.empty:
        rv['exit_date'] = pd.to_datetime(rv['exit_date'])
        rv_grouped = rv.groupby('exit_date')['net_pnl'].sum().sort_index().cumsum()
        rv_grouped.plot(title='Cumulative Net P&L (Event Day Reversion)', ylabel='USD', xlabel='Date', grid=True, legend=False, figsize=(10,4))
        plt.tight_layout()
        if save_fig:
            plt.savefig("results/figures/event_day_reversion_pnl.png")
        if show_fig:
            plt.show()
        plt.close()
    return rv, summary

# ===== 3. Buy-and-Hold Announcement to Trade Date =====
def buy_and_hold(
    events, price, 
    entry_lag=0,        # days after announcement before entering trade
    hold_days=None,     # number of days to hold (None means "to Trade Date")
    max_usd=250_000, 
    tran_cost=0.01, 
    save_fig=True, 
    show_fig=False
):
    hold_results = []
    exclusions = {'no_price': 0, 'illiquid': 0, 'too_small': 0}
    for idx, row in events.iterrows():
        ticker = row['Yahoo_Ticker']
        entry_date = row['first_tradable'] + pd.Timedelta(days=entry_lag)
        # Set exit date
        if hold_days is not None:
            exit_date = entry_date + pd.Timedelta(days=hold_days-1)
            if exit_date > row['Trade Date']:
                exit_date = row['Trade Date']
        else:
            exit_date = row['Trade Date']
        try:
            entry = price.loc[(entry_date, ticker)]
            exit_ = price.loc[(exit_date, ticker)]
        except KeyError:
            exclusions['no_price'] += 1
            continue
        adv = entry.get('ADV20', np.nan)
        if np.isnan(adv) or adv <= 0:
            exclusions['illiquid'] += 1
            continue
        px_open = entry['Open']
        px_close = exit_['Close']
        size_shares = min(int(max_usd / px_open), int(0.01 * adv))
        if size_shares < 1:
            exclusions['too_small'] += 1
            continue
        gross_pnl = (px_close - px_open) * size_shares
        total_cost = tran_cost * size_shares * 2
        net_pnl = gross_pnl - total_cost
        ret = net_pnl / (px_open * size_shares)
        hold_results.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_open': px_open,
            'exit_close': px_close,
            'shares': size_shares,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return': ret,
            'usd_invested': px_open * size_shares,
            'trade_days': (exit_date - entry_date).days + 1,
        })
    hold_df = pd.DataFrame(hold_results)
    total_pnl = hold_df['net_pnl'].sum()
    avg_return = hold_df['return'].mean()
    win = win_rate(hold_df['net_pnl'])
    sharpe = sharpe_ratio(hold_df['return'])
    sortino = sortino_ratio(hold_df['return'])
    mdd = max_drawdown(hold_df['net_pnl'].cumsum())
    summary = {
        "total_net_pnl": total_pnl,
        "avg_return": avg_return,
        "win_rate": win,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "num_trades": len(hold_df),
        "exclusions": exclusions,
    }
    print(f"SUMMARY: {summary}")
    save_summary(summary, "buy_and_hold")
    if not hold_df.empty:
        hold_df['exit_date'] = pd.to_datetime(hold_df['exit_date'])
        hold_grouped = hold_df.groupby('exit_date')['net_pnl'].sum().sort_index().cumsum()
        hold_grouped.plot(title='Cumulative Net P&L (Buy-and-Hold)', ylabel='USD', xlabel='Date', grid=True, legend=False, figsize=(10,4))
        plt.tight_layout()
        if save_fig:
            plt.savefig("results/figures/buy_and_hold_pnl.png")
        if show_fig:
            plt.show()
        plt.close()
    if hold_df.empty:
        print("No trades generated.")
        return hold_df, {}
    return hold_df, summary

# ===== 4. Hedged Momentum Strategy (Pro-Rata Allocation) =====
def hedged_momentum(events, price, spy, portfolio_gross=5_000_000, tran_cost=0.01, hold_days=5, fed_funds_rate=0.053, hedge_ratio=1.0, min_adv=1, save_fig=True, show_fig=False):
    results = []
    exclusions = {'no_price': 0, 'illiquid': 0, 'too_small': 0, 'no_spy': 0}
    CARRY_LONG = fed_funds_rate + 0.015

    for idx, row in events.iterrows():
        ticker = row['Yahoo_Ticker']
        entry_date = row['first_tradable']
        exit_date = entry_date + pd.Timedelta(days=hold_days-1)
        try:
            entry = price.loc[(entry_date, ticker)]
            exit_ = price.loc[(exit_date, ticker)]
        except KeyError:
            exclusions['no_price'] += 1
            continue

        adv = entry.get('ADV20', np.nan)
        if np.isnan(adv) or adv < min_adv:   # use meta-param min_adv
            exclusions['illiquid'] += 1
            continue
        px_open = entry['Open']
        px_close = exit_['Close']
        usd_per_trade = min(portfolio_gross / len(events), 250_000)
        size_shares = min(int(usd_per_trade / px_open), int(0.01 * adv))
        if size_shares < 1:
            exclusions['too_small'] += 1
            continue

        stock_notional = px_open * size_shares
        gross_pnl = (px_close - px_open) * size_shares
        total_cost = tran_cost * size_shares * 2
        carry_cost = stock_notional * CARRY_LONG * (hold_days / 365)

        # SPY Hedge, using hedge_ratio
        try:
            spy_open = float(spy.loc[spy['Date'] == entry_date, 'spy_open'].iloc[0])
            spy_close = float(spy.loc[spy['Date'] == exit_date, 'spy_close'].iloc[0])
        except (IndexError, KeyError):
            exclusions['no_spy'] += 1
            continue

        spy_shares = int(stock_notional * hedge_ratio / spy_open) if spy_open > 0 else 0
        spy_pnl = -spy_shares * (spy_close - spy_open) if spy_shares > 0 and not np.isnan(spy_close) else 0

        net_pnl = gross_pnl - total_cost - carry_cost + spy_pnl
        ret = net_pnl / stock_notional if stock_notional > 0 else np.nan

        # Record all meta-params
        results.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_open': px_open,
            'exit_close': px_close,
            'shares': size_shares,
            'gross_pnl': gross_pnl,
            'spy_pnl': spy_pnl,
            'hedge_ratio': hedge_ratio,   
            'carry_cost': carry_cost,
            'net_pnl': net_pnl,
            'return': ret,
            'usd_invested': stock_notional,
            'trade_days': (exit_date - entry_date).days + 1,
            'hold_days': hold_days,       
            'min_adv': min_adv,            
        })
    df = pd.DataFrame(results)
    total_pnl = df['net_pnl'].sum()
    avg_return = df['return'].mean()
    win = win_rate(df['net_pnl'])
    sharpe = sharpe_ratio(df['return'])
    sortino = sortino_ratio(df['return'])
    mdd = max_drawdown(df['net_pnl'].cumsum())
    summary = {
        "total_net_pnl": total_pnl,
        "avg_return": avg_return,
        "win_rate": win,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "num_trades": len(df),
        "exclusions": exclusions,
    }
    # PRINT summary for logs
    print(f"SUMMARY: {summary}")
    save_summary(summary, "hedged_momentum")

    # [plotting section ...]
    if not df.empty:
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        df_grouped = df.groupby('exit_date')['net_pnl'].sum().sort_index().cumsum()
        df_grouped.plot(title=f'Cumulative Net P&L (Hedged, {hold_days}-Day Hold)', ylabel='USD', xlabel='Date', grid=True, legend=False, figsize=(10,4))
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"results/figures/hedged_momentum_{hold_days}d_pnl.png")
        if show_fig:
            plt.show()
        plt.close()
    if df.empty:
        print("No trades generated.")
        return df, {}
    return df, summary

# ===== 5. Holding Period Sweep =====
def holding_period_sweep(
    events, price, spy, min_days=1, max_days=10,
    portfolio_size=5_000_000, tran_cost=0.01, fed_funds_rate=0.05,
    save_fig=True, show_fig=False
):
    import os
    os.makedirs("results/figures/sweep", exist_ok=True)
    sweep_results = []
    sweep_summaries = {}
    sweep_pnls = {}

    for hold_days in range(min_days, max_days + 1):
        strat_results = []
        exclusions = {'no_price': 0, 'illiquid': 0, 'too_small': 0, 'no_spy': 0}
        for idx, row in events.iterrows():
            ticker = row['Yahoo_Ticker']
            entry_date = row['first_tradable']
            exit_date = row['Trade Date'] - pd.Timedelta(days=1)
            # Adjust exit to max N days after entry (can't go past day before Trade Date)
            if (exit_date - entry_date).days + 1 > hold_days:
                exit_date = entry_date + pd.Timedelta(days=hold_days - 1)
            try:
                entry = price.loc[(entry_date, ticker)]
                exit_ = price.loc[(exit_date, ticker)]
                spy_entry = spy.loc[spy['Date'] == entry_date]
                spy_exit = spy.loc[spy['Date'] == exit_date]
                if spy_entry.empty or spy_exit.empty:
                    exclusions['no_spy'] += 1
                    continue
            except KeyError:
                exclusions['no_price'] += 1
                continue
            adv = entry.get('ADV20', np.nan)
            if np.isnan(adv) or adv <= 0:
                exclusions['illiquid'] += 1
                continue
            px_open = entry['Open']
            px_close = exit_['Close']
            spy_open = float(spy_entry['spy_open'].iloc[0])
            spy_close = float(spy_exit['spy_close'].iloc[0])
            size_shares = min(int(portfolio_size / len(events) / px_open), int(0.01 * adv))
            if size_shares < 1:
                exclusions['too_small'] += 1
                continue
            gross_pnl_stock = (px_close - px_open) * size_shares
            gross_pnl_spy = -(spy_close - spy_open) * size_shares * (px_open / spy_open)
            carry_stock = (px_open * size_shares) * ((fed_funds_rate + 0.015) / 252) * (exit_date - entry_date).days
            carry_spy = (spy_open * size_shares) * ((fed_funds_rate + 0.015) / 252) * (exit_date - entry_date).days
            total_cost = tran_cost * size_shares * 2
            net_pnl = gross_pnl_stock + gross_pnl_spy - total_cost - carry_stock - carry_spy
            ret = net_pnl / (px_open * size_shares)
            strat_results.append({
                'ticker': ticker,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'shares': size_shares,
                'entry_open': px_open,
                'exit_close': px_close,
                'spy_open': spy_open,
                'spy_close': spy_close,
                'gross_pnl_stock': gross_pnl_stock,
                'gross_pnl_spy': gross_pnl_spy,
                'total_cost': total_cost,
                'carry_stock': carry_stock,
                'carry_spy': carry_spy,
                'net_pnl': net_pnl,
                'return': ret,
                'usd_invested': px_open * size_shares,
                'trade_days': (exit_date - entry_date).days + 1,
                'hold_days': hold_days,
            })
        strat_df = pd.DataFrame(strat_results)
        # Performance metrics for this holding period
        total_pnl = strat_df['net_pnl'].sum()
        avg_return = strat_df['return'].mean()
        win = win_rate(strat_df['net_pnl'])
        sharpe = sharpe_ratio(strat_df['return'])
        sortino = sortino_ratio(strat_df['return'])
        mdd = max_drawdown(strat_df['net_pnl'].cumsum())
        sweep_summaries[hold_days] = {
            "total_net_pnl": total_pnl,
            "avg_return": avg_return,
            "win_rate": win,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": mdd,
            "num_trades": len(strat_df),
            "exclusions": exclusions,
        }

        # PRINT summary for logs
        print(f"SUMMARY: {sweep_summaries}")
        save_summary(sweep_summaries, "holding_period_sweep")

        # Save the cumulative PnL curve for this holding period
        equity_curve = strat_df.sort_values('exit_date').set_index('exit_date')['net_pnl'].cumsum()
        sweep_pnls[hold_days] = equity_curve
        sweep_results.extend(strat_results)
        # Save curve plot for this period
        if save_fig and not equity_curve.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(equity_curve.index, equity_curve.values, label=f'{hold_days} days')
            plt.title(f'Cumulative Net P&L ({hold_days}-Day Hold)')
            plt.ylabel('USD')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/figures/sweep/holding_period_{hold_days}d_cum_pnl.png")
            plt.close()

    # Plot all curves on one summary plot
    plt.figure(figsize=(12, 6))
    for hold_days, curve in sweep_pnls.items():
        if not curve.empty:
            plt.plot(curve.index, curve.values, label=f'{hold_days} days')
    plt.title('Cumulative Net P&L by Holding Period (Sweep)')
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig("results/figures/sweep/holding_period_sweep_cum_pnl.png")
    if show_fig:
        plt.show()
    plt.close()

    # Return: all trades as a DataFrame, summaries, and curves for the notebook/report
    return pd.DataFrame(sweep_results), sweep_summaries, sweep_pnls