# =========================================================================
# DO NOT EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

import pandas as pd
import numpy as np
import joblib

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)


try:
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data 
except ImportError:
    print("FATAL ERROR: Could not import backtest_engine.py. Check your directory structure.")
    exit()


print("\n--- LOADING TRAINING DATA ---")
df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

# =========================================================================
# EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

# --- CONFIGURATION (Participants can adjust these) ---
TEAM_NAME = "my_team_name"
SUBMISSION_NAME = f'{TEAM_NAME}_submission.joblib'
INITIAL_CAPITAL = 10000.0
FAST_WINDOW = 20
SLOW_WINDOW = 50
N_DAYS_PREDICT = 2
# ---------------------------------------------------

# TODO: Implement additional features as needed
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.rolling(window=FAST_WINDOW).mean()
)
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.rolling(window=SLOW_WINDOW).mean()
)
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']


# TARGET VARIABLE DEFINITION
df['Future_Return'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)
)

df.dropna(inplace=True)

# TODO: FEATURE SELECTION - UPDATE THIS LIST WITH YOUR FEATURES
FEATURE_COLS = ['MA_Difference']

# =========================================================================
# MODEL DEFINITION SECTION - CHOOSE YOUR ML APPROACH
# =========================================================================

print(f"\n--- MODEL TRAINING ---")

# Split data
train_size = int(len(df) * 0.80)
X = df[FEATURE_COLS]
y = df['Future_Return']

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# IMPORT YOUR MODEL HERE and replace the code below
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train_scaled, y_train)

# TRADING SIGNAL GENERATION

print("\n--- GENERATING TRADING SIGNALS ---")

df_local_test = df.iloc[train_size:].copy()
predictions = model.predict(X_test_scaled)
df_local_test['Predicted_Return'] = predictions
df_local_test['Signal'] = np.where(df_local_test['Predicted_Return'] > 0, 1, 0)

# =========================================================================
# DO NOT EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

print("\n--- RUNNING BACKTEST ---")

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    run_single_stock_analysis(
        df_ticker_data, 
        ticker, 
        INITIAL_CAPITAL, 
        TEAM_NAME, 
        strategy_file_path=CURRENT_STRATEGY_PATH
    )

# =========================================================================
# MODEL SAVING (DO NOT MODIFY)
# =========================================================================

joblib.dump(model, SUBMISSION_NAME)
print(f"\n SUBMISSION READY: Model saved as {SUBMISSION_NAME}")