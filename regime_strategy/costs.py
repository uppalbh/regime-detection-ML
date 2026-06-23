
def compute_slippage_series(close, lookback=20):
    """Rolling volatility-scaled slippage (per side)."""
    daily_vol = close.pct_change().rolling(lookback).std().fillna(
        close.pct_change().std()
    )
    return SLIPPAGE_SCALE * daily_vol


def compute_atr_series(close, lookback=14):
    """Simple close-to-close ATR proxy."""
    return close.pct_change().abs().rolling(lookback).mean().fillna(
        close.pct_change().abs().mean()
    )

