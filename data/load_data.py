"""
Data Loading Module with Multiple Fallback Options
===================================================

This module provides robust data loading with multiple fallback methods:
1. CSV cache (fastest, works offline)
2. yfinance (recommended - actively maintained)
3. pandas_datareader (backup - often unreliable)

WHY pandas-datareader often fails:
----------------------------------
- Yahoo Finance deprecated their free API that pandas-datareader used
- The get_data_yahoo() function is no longer maintained
- Yahoo Finance changed their API structure, breaking compatibility
- Solution: Use yfinance instead, which is actively maintained and works with Yahoo Finance

RECOMMENDED SETUP:
------------------
1. Install yfinance: pip install yfinance
2. Run once to cache data: python data/download_and_cache_data.py
3. Future runs will use cached CSV (fast and offline-capable)
"""

import pandas as pd
import os

# --- CONSTANTS ---
TICKERS = ['MSFT', 'AAPL', 'AMZN', 'JPM'] # List of stocks for the portfolio
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2023-12-31'
DATA_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'stock_data_cache.csv')
# -----------------


def _load_from_yfinance(tickers, start_date, end_date):
    """
    Try loading data using yfinance library.
    This is the recommended method for Yahoo Finance data.
    """
    try:
        import yfinance as yf
        print("-> Attempting to load data using yfinance...")
        
        # Download data with retry logic
        try:
            data = yf.download(
                tickers, 
                start=start_date, 
                end=end_date, 
                interval='1d',
                progress=False,
                auto_adjust=True,
                group_by='ticker'
            )
        except Exception as e:
            # If batch download fails, try individual tickers
            print(f"  -> Batch download failed, trying individual tickers...")
            dfs = {}
            for ticker in tickers:
                try:
                    ticker_data = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        interval='1d',
                        progress=False,
                        auto_adjust=True
                    )
                    if not ticker_data.empty and 'Close' in ticker_data.columns:
                        dfs[ticker] = ticker_data['Close']
                        print(f"  -> Loaded {ticker}")
                except Exception as ticker_error:
                    print(f"  -> Failed to load {ticker}: {ticker_error}")
                    continue
            
            if not dfs:
                raise ValueError("No data loaded from any ticker")
            
            data = pd.DataFrame(dfs)
        
        # Handle different data structures
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                # Single ticker case
                data = data[['Close']]
                data.columns = [tickers[0] if len(tickers) == 1 else 'Close']
            elif len(data.columns) > 0:
                # Multi-ticker case - columns should be ticker names
                pass
            else:
                raise ValueError("Unexpected data structure from yfinance")
        
        if data.empty:
            raise ValueError("No data returned from yfinance.")
        
        # Melt the DataFrame to create a MultiIndex (Date, Ticker) format
        df = data.stack().to_frame(name='Close')
        df.index.names = ['Date', 'Ticker']
        return df
    except ImportError:
        print("-> yfinance not available, trying alternative...")
        print("  -> Install with: pip install yfinance")
        return None
    except Exception as e:
        print(f"-> yfinance failed: {e}")
        print("  -> Common issues:")
        print("     - Network connectivity problems")
        print("     - Yahoo Finance API rate limiting")
        print("     - Invalid date range or ticker symbols")
        print("  -> Try: pip install --upgrade yfinance")
        return None


def _load_from_pandas_datareader(tickers, start_date, end_date):
    """
    Try loading data using pandas_datareader library.
    
    NOTE: get_data_yahoo() is deprecated and often fails due to Yahoo Finance API changes.
    This function tries alternative sources like 'stooq' which may be more reliable.
    """
    try:
        import pandas_datareader.data as web
        print("-> Attempting to load data using pandas_datareader (stooq source)...")
        
        # Try stooq as it's more reliable than Yahoo
        dfs = {}
        for ticker in tickers:
            try:
                # Try stooq first (more reliable)
                data = web.DataReader(ticker, 'stooq', start=start_date, end=end_date)
                if not data.empty and 'Close' in data.columns:
                    dfs[ticker] = data['Close']
                    print(f"  -> Loaded {ticker} from stooq")
                else:
                    raise ValueError("No Close column in data")
            except Exception as e1:
                # Fallback: try Yahoo (though it's often broken)
                try:
                    print(f"  -> stooq failed for {ticker}, trying Yahoo...")
                    data = web.get_data_yahoo(ticker, start=start_date, end=end_date)
                    if not data.empty:
                        dfs[ticker] = data['Close']
                        print(f"  -> Loaded {ticker} from Yahoo")
                except Exception as e2:
                    print(f"  -> Failed to load {ticker}: stooq={e1}, yahoo={e2}")
                    continue
        
        if not dfs:
            raise ValueError("No data loaded from pandas_datareader.")
        
        # Combine all tickers into a DataFrame
        combined = pd.DataFrame(dfs)
        
        # Melt the DataFrame to create a MultiIndex (Date, Ticker) format
        df = combined.stack().to_frame(name='Close')
        df.index.names = ['Date', 'Ticker']
        return df
    except ImportError:
        print("-> pandas_datareader not available, trying CSV...")
        return None
    except Exception as e:
        print(f"-> pandas_datareader failed: {e}")
        print("  -> Note: pandas_datareader's Yahoo source is often broken.")
        print("  -> Consider using yfinance or CSV cache instead.")
        return None


def _load_from_csv(csv_file):
    """Load data from CSV cache file."""
    try:
        if not os.path.exists(csv_file):
            return None
        
        print(f"-> Loading data from CSV cache: {csv_file}")
        df = pd.read_csv(csv_file, index_col=[0, 1], parse_dates=[0])
        df.index.names = ['Date', 'Ticker']
        print(f"-> Successfully loaded {len(df)} price observations from CSV.")
        return df
    except Exception as e:
        print(f"-> CSV load failed: {e}")
        return None


def _save_to_csv(df, csv_file):
    """Save DataFrame to CSV for future use."""
    try:
        df.to_csv(csv_file)
        print(f"-> Data cached to {csv_file} for future use.")
    except Exception as e:
        print(f"-> Warning: Could not save cache file: {e}")


def load_training_data(tickers=TICKERS, start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE, 
                      use_cache=True, save_cache=True):
    """
    Downloads historical close price data for a list of tickers and returns 
    a single Pandas DataFrame with a MultiIndex (Date, Ticker).
    
    Tries multiple data sources in order:
    1. CSV cache (if use_cache=True)
    2. yfinance (recommended - most reliable)
    3. pandas_datareader (stooq source, then Yahoo as fallback)
    4. CSV cache as last resort
    
    Note: pandas_datareader's Yahoo source is often broken due to API changes.
    yfinance is the recommended library for Yahoo Finance data.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    use_cache : bool
        If True, try to load from CSV cache first
    save_cache : bool
        If True, save downloaded data to CSV cache
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with MultiIndex (Date, Ticker) and 'Close' column
    """
    print(f"-> Fetching training data for {len(tickers)} stocks ({start_date} to {end_date})...")
    
    # Try CSV cache first if enabled
    if use_cache:
        df = _load_from_csv(DATA_CACHE_FILE)
        if df is not None:
            return df
    
    # Try yfinance
    df = _load_from_yfinance(tickers, start_date, end_date)
    if df is not None:
        if save_cache:
            _save_to_csv(df, DATA_CACHE_FILE)
        print(f"-> Successfully loaded {len(df)} price observations for {len(tickers)} assets.")
        return df
    
    # Try pandas_datareader
    df = _load_from_pandas_datareader(tickers, start_date, end_date)
    if df is not None:
        if save_cache:
            _save_to_csv(df, DATA_CACHE_FILE)
        print(f"-> Successfully loaded {len(df)} price observations for {len(tickers)} assets.")
        return df
    
    # Last resort: try CSV even if use_cache was False
    df = _load_from_csv(DATA_CACHE_FILE)
    if df is not None:
        print("-> Using cached data as fallback.")
        return df
    
    # All methods failed
    print("FATAL DATA ERROR: Could not load portfolio data from any source.")
    print("Please ensure:")
    print("  1. You have internet connection")
    print("  2. Install one of: pip install yfinance OR pip install pandas-datareader")
    print("  3. Or provide a CSV file at:", DATA_CACHE_FILE)
    return pd.DataFrame()


if __name__ == '__main__':
    df_test = load_training_data()
    print(df_test.head(10))