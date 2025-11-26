import pandas as pd
import numpy as np
import joblib
# Import the K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
import sys
import os

# --- CRITICAL PATH FIX ---
# This ensures the bot can import modules from the project root (like backtest_engine.py).
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assumes structure: /Project_Root/strategies/knn_cash_cow/strategy.py
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) 
sys.path.append(project_root)
# -------------------------

# --- IMPORTANT: Import the fixed backtesting engine and data loader ---
try:
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data 
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules. Check project structure and imports. Error: {e}")
    exit()
# ------------------------------------------------------------------

# --- CONFIGURATION (Participants can adjust these) ---
FAST_WINDOW = 20
SLOW_WINDOW = 50
N_DAYS_PREDICT = 5       
SUBMISSION_NAME = 'my_team_name_knn_submission.joblib'
INITIAL_CAPITAL = 10000.0 
# ---------------------------------------------------

# 1. LOAD TRAINING DATA
df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

# =========================================================================
# 🎯 SECTION A: FEATURE ENGINEERING (LEVER 1: Add features)
# =========================================================================
print("\n--- 1. FEATURE ENGINEERING ---")

# CURRENT SIMPLE FEATURES 
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']

# Create the Target Variable: CONVERT TO BINARY CLASSIFICATION LABEL
# Y = 1 (Up) if the future return is positive, 0 (Down/Flat) otherwise.
df['Future_Return_Class'] = np.where(df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)) > 0, 1, 0)

df.dropna(inplace=True)

# 2. SPLIT & STANDARDIZATION
# LEVER 2: Participants MUST update this list if they add new features!
FEATURE_COLS = ['MA_Difference'] 
X = df[FEATURE_COLS]
# Note: We use the new binary target column for classification
y = df['Future_Return_Class'] 

train_size = int(len(df) * 0.80)

# SCALING IS CRUCIAL FOR KNN!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[:train_size])
y_train = y.iloc[:train_size]

df_local_test = df.iloc[train_size:].copy() 
team_name = SUBMISSION_NAME.split('_submission')[0]

# =========================================================================
# 🎯 SECTION B: TRAIN CLASSIFICATION MODEL (LEVER 3: Tune the model)
# =========================================================================
print(f"\n--- 2. MODEL TRAINING ({len(X_train_scaled)} samples) ---")

# Key tuning levers: n_neighbors (K) and weights (uniform or distance).
model = KNeighborsClassifier(
    n_neighbors=15, # Try adjusting K (must be an odd number to avoid ties)
    weights='distance', # Weights points by the inverse of their distance
    algorithm='auto'
).fit(X_train_scaled, y_train)

# =========================================================================
# 🎯 SECTION C: EXECUTION (FIXED - Uses the Backtest Engine)
# =========================================================================

# Get the full path of the current strategy file to save plots correctly
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)

# 3. RUN INDIVIDUAL STOCK ANALYSIS
print("\n--- 3. Running Individual Stock Timing Analysis ---")

X_test_scaled = scaler.transform(df_local_test[FEATURE_COLS])

# KNN predicts the class (0 or 1) directly
predicted_class = model.predict(X_test_scaled) 

df_local_test['Predicted_Return'] = predicted_class # Store the class here for reference
# Signal: The direct predicted class (1 for up, 0 for down/flat)
df_local_test['Signal'] = predicted_class

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    # Call the fixed backtesting engine function
    run_single_stock_analysis(df_ticker_data, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH)

# 4. SUBMIT (SAVE) THE FINAL MODEL
joblib.dump(model, SUBMISSION_NAME)
print(f"\nSUBMISSION READY: Model saved as {SUBMISSION_NAME}")