import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- FIX FOR DIRECT EXECUTION ---
import sys
import os
# Add the project root directory to the path to resolve imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------

# IMPORT MODULES
from data.load_data import load_training_data 

# --- CONFIGURATION ---
FAST_WINDOW = 20
SLOW_WINDOW = 50
N_DAYS_PREDICT = 5       
SUBMISSION_NAME = 'my_team_name_mlp_submission.joblib'
INITIAL_CAPITAL = 10000.0 
# ---------------------

def run_portfolio_backtest(df_test: pd.DataFrame, model, scaler: StandardScaler, initial_capital: float, team_name: str):
    """
    Runs a Transaction-Based Backtest (Shares/Cash tracking) using EWP filter with DAILY REBALANCING.
    This method guarantees numerical stability and active trading.
    """
    print(f"\n--- Running Local Transaction-Based Backtest (Starting with ${initial_capital:,.2f}) ---")
    
    FEATURE_COLS = ['MA_Difference']
    
    # 1. PREDICT FUTURE RETURNS
    X_test_scaled = scaler.transform(df_test[FEATURE_COLS])
    df_test['Predicted_Return'] = model.predict(X_test_scaled)

    # 2. SETUP FOR TRANSACTION TRACKING
    
    # Unstack the data so we have a single date index and columns for each ticker's price
    df_prices = df_test['Close'].unstack(level='Ticker').copy()
    
    # Create the prediction signal (1 = Buy, 0 = Hold/Do Nothing)
    df_test['Signal'] = np.where(df_test['Predicted_Return'] > 0, 1, 0)
    signal_df = df_test['Signal'].unstack(level='Ticker')

    # Initialize tracking variables (Index: Date)
    daily_trade_signal = signal_df.shift(1).fillna(0) # Signal from day T-1 is used for trade on day T
    cash_history = pd.Series(index=df_prices.index, dtype=float)
    portfolio_equity = pd.Series(index=df_prices.index, dtype=float)
    
    # Shares held (DataFrame indexed by Date, Columns by Ticker)
    shares_held = pd.DataFrame(0.0, index=df_prices.index, columns=df_prices.columns)
    
    # Initialize the portfolio
    current_cash = initial_capital
    
    # 3. RUN SIMULATION DAY-BY-DAY (The Active Trading Loop)
    for i, date in enumerate(df_prices.index):
        
        # --- A. LIQUIDATION AND REBALANCING ---
        if i > 0:
            prev_date = df_prices.index[i - 1]
            
            # 1. LIQUIDATE: Calculate the total value of yesterday's holdings at today's price
            prev_shares = shares_held.loc[prev_date]
            liquidation_value = (prev_shares * df_prices.loc[date]).sum()
            
            # 2. UPDATE CASH: Portfolio value = Cash (from yesterday) + Liquidation Value
            current_cash = cash_history.loc[prev_date] + liquidation_value
            
            # 3. RESET SHARES: We start today with 0 shares (they were liquidated)
            shares_held.loc[date] = 0.0
        
        # --- B. TRADING (Allocate based on today's signal) ---
        
        buy_signals = daily_trade_signal.loc[date][daily_trade_signal.loc[date] == 1]
        num_buys = len(buy_signals)
        
        if num_buys > 0:
            # Invest all available capital equally into the selected stocks (EWP)
            investment_amount = current_cash
            capital_per_stock = investment_amount / num_buys
            
            for ticker in buy_signals.index:
                price = df_prices.loc[date, ticker]
                
                # Ensure price is valid before division
                if not pd.isna(price) and price > 0:
                    shares = capital_per_stock / price
                    shares_held.loc[date, ticker] += shares
                    current_cash -= capital_per_stock
        
        # --- C. VALUE PORTFOLIO ---
        
        # Value of stock holdings (Shares * Current Price)
        stock_value = (shares_held.loc[date] * df_prices.loc[date]).sum()
        
        # Total Portfolio Value
        portfolio_value = stock_value + current_cash
        
        # Record history
        cash_history.loc[date] = current_cash
        portfolio_equity.loc[date] = portfolio_value

    # 4. METRICS & PLOTTING 
    
    # Calculate daily returns from the equity curve (stable)
    daily_returns = portfolio_equity.pct_change().dropna()
    final_balance = portfolio_equity.iloc[-1]
    
    DAYS_IN_YEAR = 252
    mean_return = daily_returns.mean() * DAYS_IN_YEAR
    std_dev_return = daily_returns.std() * np.sqrt(DAYS_IN_YEAR)
    sharpe_ratio = mean_return / std_dev_return if std_dev_return != 0 else 0
    
    print(f"Final Portfolio Balance: ${final_balance:,.2f}") 
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.4f}")
    
    # Save the plot to a file
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_equity.index, portfolio_equity.values, label=f'Portfolio Equity (Sharpe: {sharpe_ratio:.2f})')
    plt.title(f'Local Backtest: {team_name} Portfolio Allocation Performance')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(y=initial_capital, color='r', linestyle='-', label='Initial Capital')
    plt.legend()
    
    PLOT_FILENAME = f"backtest_equity_{team_name}.png"
    plt.savefig(PLOT_FILENAME)
    print(f"Plot saved successfully as {PLOT_FILENAME}")
    plt.close()


# --- MAIN EXECUTION CODE ---

# 1. LOAD TRAINING DATA
df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

# 2. FEATURE ENGINEERING & TARGET CREATION
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']

df['Future_Return'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT))

df.dropna(inplace=True)

# 3. SPLIT & STANDARDIZATION
FEATURE_COLS = ['MA_Difference']
X = df[FEATURE_COLS]
y = df['Future_Return']

train_size = int(len(df) * 0.80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[:train_size])
y_train = y.iloc[:train_size]

df_local_test = df.iloc[train_size:].copy() 
team_name = SUBMISSION_NAME.split('_submission')[0]

# 4. TRAIN REGRESSION MODEL
print(f"Training MLP Regressor with {len(X_train_scaled)} samples...")
model = MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50)).fit(X_train_scaled, y_train)

# 5. RUN BACKTEST AND PLOT
run_portfolio_backtest(df_local_test, model, scaler, INITIAL_CAPITAL, team_name)

# 6. SUBMIT (SAVE) THE FINAL MODEL
joblib.dump(model, SUBMISSION_NAME)
print(f"\nSUBMISSION READY: Model saved as {SUBMISSION_NAME}")