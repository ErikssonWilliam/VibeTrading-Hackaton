"""
Technical Indicators Library for Trading Strategies
===================================================
This module provides pandas-compatible functions for computing common technical indicators.
All functions work with pandas Series and can be easily used with groupby operations.

Usage Example:
    import pandas as pd
    from technical_indicators import rsi, macd, bollinger_bands
    
    # For a single ticker
    df['RSI'] = rsi(df['Close'], period=14)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
    
    # For multi-index DataFrame (with Ticker and Date)
    df['RSI'] = df.groupby(level='Ticker')['Close'].transform(lambda x: rsi(x, period=14))
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Simple Moving Average (SMA)
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=20
        Number of periods for the moving average
    
    Returns:
    --------
    pd.Series
        Simple Moving Average values
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA)
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=20
        Number of periods for the moving average
    
    Returns:
    --------
    pd.Series
        Exponential Moving Average values
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)
    RSI ranges from 0 to 100. Values above 70 indicate overbought, below 30 indicate oversold.
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=14
        Number of periods for RSI calculation
    
    Returns:
    --------
    pd.Series
        RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Moving Average Convergence Divergence (MACD)
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    fast : int, default=12
        Period for fast EMA
    slow : int, default=26
        Period for slow EMA
    signal : int, default=9
        Period for signal line EMA
    
    Returns:
    --------
    tuple of pd.Series
        (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(series, period=fast)
    ema_slow = ema(series, period=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, period=signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=20
        Number of periods for the moving average
    std_dev : float, default=2.0
        Number of standard deviations for the bands
    
    Returns:
    --------
    tuple of pd.Series
        (Upper Band, Middle Band (SMA), Lower Band)
    """
    middle_band = sma(series, period=period)
    std = series.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> tuple:
    """
    Stochastic Oscillator (%K and %D)
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    k_period : int, default=14
        Period for %K calculation
    d_period : int, default=3
        Period for %D (smoothing of %K)
    
    Returns:
    --------
    tuple of pd.Series
        (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR)
    Measures market volatility
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default=14
        Number of periods for ATR calculation
    
    Returns:
    --------
    pd.Series
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX)
    Measures trend strength (0-100, higher = stronger trend)
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default=14
        Number of periods for ADX calculation
    
    Returns:
    --------
    pd.Series
        ADX values
    """
    atr_values = atr(high, low, close, period)
    
    # Calculate +DM and -DM
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smooth the DMs
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_values)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_values)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Williams %R
    Momentum indicator ranging from -100 to 0. Values above -20 indicate overbought, below -80 indicate oversold.
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default=14
        Number of periods for calculation
    
    Returns:
    --------
    pd.Series
        Williams %R values
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return wr


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI)
    Values above +100 indicate overbought, below -100 indicate oversold.
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    period : int, default=20
        Number of periods for calculation
    
    Returns:
    --------
    pd.Series
        CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period=period)
    mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume (OBV)
    Requires volume data
    
    Parameters:
    -----------
    close : pd.Series
        Close prices
    volume : pd.Series
        Volume data
    
    Returns:
    --------
    pd.Series
        OBV values
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum Indicator
    Measures the rate of change in price
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=10
        Number of periods to look back
    
    Returns:
    --------
    pd.Series
        Momentum values
    """
    return series.diff(periods=period)


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change (ROC)
    Percentage change in price over a period
    
    Parameters:
    -----------
    series : pd.Series
        Price series (typically Close prices)
    period : int, default=10
        Number of periods to look back
    
    Returns:
    --------
    pd.Series
        ROC values (as percentage)
    """
    return series.pct_change(periods=period) * 100


def price_channels(high: pd.Series, low: pd.Series, period: int = 20) -> tuple:
    """
    Price Channels (Donchian Channels)
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    period : int, default=20
        Number of periods for calculation
    
    Returns:
    --------
    tuple of pd.Series
        (Upper Channel, Lower Channel, Middle Channel)
    """
    upper_channel = high.rolling(window=period).max()
    lower_channel = low.rolling(window=period).min()
    middle_channel = (upper_channel + lower_channel) / 2
    
    return upper_channel, lower_channel, middle_channel


def parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, 
                  af_start: float = 0.02, af_increment: float = 0.02, 
                  af_max: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse)
    Trend-following indicator
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    af_start : float, default=0.02
        Acceleration factor start value
    af_increment : float, default=0.02
        Acceleration factor increment
    af_max : float, default=0.2
        Maximum acceleration factor
    
    Returns:
    --------
    pd.Series
        Parabolic SAR values
    """
    psar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)
    ep = pd.Series(index=close.index, dtype=float)  # Extreme Point
    af = pd.Series(index=close.index, dtype=float)  # Acceleration Factor
    
    # Initialize
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
    ep.iloc[0] = high.iloc[0]
    af.iloc[0] = af_start
    
    for i in range(1, len(close)):
        if trend.iloc[i-1] == 1:  # Uptrend
            psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i])
            
            if high.iloc[i] > ep.iloc[i-1]:
                ep.iloc[i] = high.iloc[i]
                af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
            else:
                ep.iloc[i] = ep.iloc[i-1]
                af.iloc[i] = af.iloc[i-1]
            
            if low.iloc[i] < psar.iloc[i]:
                trend.iloc[i] = -1
                psar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = low.iloc[i]
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = 1
        else:  # Downtrend
            psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i])
            
            if low.iloc[i] < ep.iloc[i-1]:
                ep.iloc[i] = low.iloc[i]
                af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
            else:
                ep.iloc[i] = ep.iloc[i-1]
                af.iloc[i] = af.iloc[i-1]
            
            if high.iloc[i] > psar.iloc[i]:
                trend.iloc[i] = 1
                psar.iloc[i] = ep.iloc[i-1]
                ep.iloc[i] = high.iloc[i]
                af.iloc[i] = af_start
            else:
                trend.iloc[i] = -1
    
    return psar

