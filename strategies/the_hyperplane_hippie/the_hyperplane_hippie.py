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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)
# -------------------------------

try:
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data 
except ImportError:
    print("FATAL ERROR: Could not import backtest_engine.py. Check your directory structure.")
    exit()
# ------------------------------------------------------------------

# --- CONFIGURATION (Participants can adjust these) ---
FAST_WINDOW = 20
SLOW_WINDOW = 50
N_DAYS_PREDICT = 20       # LEVER 1: Adjust prediction horizon
SUBMISSION_NAME = 'my_team_name_mlp_submission.joblib'
INITIAL_CAPITAL = 10000.0 
# ---------------------------------------------------

# 1. LOAD TRAINING DATA
df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

# =========================================================================
# 🎯 SECTION A: FEATURE ENGINEERING (LEVER 2: Add features to generate Alpha)
# =========================================================================
print("\n--- 1. FEATURE ENGINEERING ---")

# CURRENT SIMPLE FEATURES (Trend Following)
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']

# TODO: Participants should add new, leading indicators here (e.g., RSI, ATR, volume analysis).

# Create the Target Variable (Future Return)
df['Future_Return'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT))

df.dropna(inplace=True)

# 2. SPLIT & STANDARDIZATION
# IMPORTANT: Participants MUST update this list if they add new features!
FEATURE_COLS = ['MA_Difference'] 
X = df[FEATURE_COLS]
y = df['Future_Return']

train_size = int(len(df) * 0.80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[:train_size])
y_train = y.iloc[:train_size]

df_local_test = df.iloc[train_size:].copy() 
team_name = SUBMISSION_NAME.split('_submission')[0]

# =========================================================================
# 🎯 SECTION B: TRAIN REGRESSION MODEL (LEVER 3: Tune the model)
# =========================================================================
print(f"\n--- 2. MODEL TRAINING ({len(X_train_scaled)} samples) ---")

# Participants should adjust hidden_layer_sizes, max_iter, or activation
model = MLPRegressor(
    random_state=42, 
    max_iter=500, # Try increasing for better convergence
    hidden_layer_sizes=(150, 75) # Try (150, 75) or (50, 50, 25)
).fit(X_train_scaled, y_train)

# =========================================================================
# SECTION C: EXECUTION DO NOT EDIT THIS (FIXED - Uses the Backtest Engine)
# =========================================================================

# 3. RUN INDIVIDUAL STOCK ANALYSIS
print("\n--- 3. Running Individual Stock Timing Analysis ---")

X_test_scaled = scaler.transform(df_local_test[FEATURE_COLS])
df_local_test['Predicted_Return'] = model.predict(X_test_scaled)
df_local_test['Signal'] = np.where(df_local_test['Predicted_Return'] > 0, 1, 0)

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    # Call the fixed backtesting engine function
    run_single_stock_analysis(df_ticker_data, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH)

# 4. SUBMIT (SAVE) THE FINAL MODEL
joblib.dump(model, SUBMISSION_NAME)
print(f"\nSUBMISSION READY: Model saved as {SUBMISSION_NAME}")