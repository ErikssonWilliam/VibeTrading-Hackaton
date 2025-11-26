import pandas as pd
import yfinance as yf

# --- CONSTANTS ---
TICKERS = ['MSFT', 'AAPL', 'AMZN', 'JPM'] # List of stocks for the portfolio
TRAIN_START_DATE = '2018-01-01'
TRAIN_END_DATE = '2023-12-31' 
# -----------------

def load_training_data(tickers=TICKERS, start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE):
    """
    Downloads historical close price data for a list of tickers and returns 
    a single Pandas DataFrame with a MultiIndex (Date, Ticker).
    """
    print(f"-> Fetching training data for {len(tickers)} stocks...")
    try:
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            interval='1d',
            progress=False,
            auto_adjust=True 
        )['Close'] 
        
        if data.empty:
             raise ValueError("No data returned for the specified range/tickers.")
        
        # Melt the DataFrame to create a MultiIndex (Date, Ticker) format
        df = data.stack().to_frame(name='Close')
        df.index.names = ['Date', 'Ticker']
             
        print(f"-> Successfully loaded {len(df)} price observations for {len(tickers)} assets.")
        return df
        
    except Exception as e:
        print(f"FATAL DATA ERROR: Could not load portfolio data: {e}")
        return pd.DataFrame() 

if __name__ == '__main__':
    df_test = load_training_data()
    print(df_test.head(10))