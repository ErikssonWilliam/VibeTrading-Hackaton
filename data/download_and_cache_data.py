"""
Helper script to download stock data and save it to CSV for offline use.
Run this script once to create a cache file that can be used when online sources fail.

Usage:
    python data/download_and_cache_data.py
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from data.load_data import load_training_data, DATA_CACHE_FILE

if __name__ == '__main__':
    print("=" * 60)
    print("Stock Data Downloader and Cache Creator")
    print("=" * 60)
    print()
    print("This script will attempt to download stock data and save it")
    print("to a CSV file for offline use.")
    print()
    
    # Try to load data (will attempt all methods)
    df = load_training_data(use_cache=False, save_cache=True)
    
    if df.empty:
        print()
        print("ERROR: Could not download data.")
        print()
        print("Please try one of the following:")
        print("  1. Install yfinance: pip install yfinance")
        print("  2. Install pandas-datareader: pip install pandas-datareader")
        print("  3. Check your internet connection")
        print("  4. Manually create a CSV file at:", DATA_CACHE_FILE)
        print()
        sys.exit(1)
    else:
        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Data has been cached to: {DATA_CACHE_FILE}")
        print(f"Total records: {len(df)}")
        print(f"Tickers: {df.index.get_level_values('Ticker').unique().tolist()}")
        print()
        print("You can now use the data loader with use_cache=True")
        print("or even without internet connection (if CSV exists).")
        print()

