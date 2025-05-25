from strategies import post_announcement_momentum, event_day_reversion, buy_and_hold, hedged_momentum
from utils import save_summary


def sweep_post_announcement(events, price, hold_periods, **kwargs):
    sweep_metrics = {}
    for N in hold_periods:
        bt, summary = post_announcement_momentum(
            events, price,
            hold_days=N,
            **kwargs
        )
        sweep_metrics[str(N)] = summary  # Ensure key is a string
    save_summary(sweep_metrics, "post_announcement_momentum_sweep")
    return sweep_metrics



def sweep_event_day_reversion(events, price, thresholds, hold_periods, **kwargs):
    sweep_metrics = {}
    for th in thresholds:
        for N in hold_periods:
            rv, summary = event_day_reversion(
                events, price,
                threshold=th, hold_days=N,  # <-- Add hold_days param
                **kwargs
            )
            sweep_metrics[f"{th}_{N}"] = summary  # Use string key
    save_summary(sweep_metrics, "event_day_reversion_sweep")
    return sweep_metrics



def sweep_buy_and_hold(events, price, entry_lags, hold_periods, **kwargs):
    sweep_metrics = {}
    for lag in entry_lags:
        for N in hold_periods:
            bh, summary = buy_and_hold(
                events, price,
                entry_lag=lag, hold_days=N,
                **kwargs
            )
            sweep_metrics[f"{lag}_{N}"] = summary  # Use string key
    save_summary(sweep_metrics, "buy_and_hold_sweep")
    return sweep_metrics


def sweep_hedged_momentum(
    events, price, spy,
    hedge_ratios=[1.0],
    hold_periods=[5],
    min_advs=[1],
    **kwargs
):
    sweep_metrics = {}
    for hr in hedge_ratios:
        for N in hold_periods:
            for ma in min_advs:
                df, summary = hedged_momentum(
                    events, price, spy,
                    hedge_ratio=hr,
                    hold_days=N,
                    min_adv=ma,
                    **kwargs
                )
                sweep_metrics[f"{hr}_{N}_{ma}"] = summary  # Use string key
    save_summary(sweep_metrics, "hedged_momentum_sweep")
    return sweep_metrics