# =========================================================================
# Start of file. Do not edit this section.
# RANDOM FOREST STRATEGY IMPLEMENTATION
# =========================================================================
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

try:
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data 
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules. Check project structure and imports. Error: {e}")
    exit()

df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

# =========================================================================
# EDIT FROM THIS POINT DOWNWARDS
# =========================================================================
# --- CONFIGURATION (Participants can adjust these) ---
FAST_WINDOW = 20
SLOW_WINDOW = 50
N_DAYS_PREDICT = 5       
SUBMISSION_NAME = 'my_team_name_rfr_submission.joblib' 
INITIAL_CAPITAL = 10000.0 
# ---------------------------------------------------


# TODO: Implement additional features as needed 
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']

# Create the Target Variable: CONVERT TO BINARY CLASSIFICATION LABEL
# Y = 1 (Up) if the future return is positive, 0 (Down/Flat) otherwise.
df['Future_Return_Class'] = np.where(df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)) > 0, 1, 0)

df.dropna(inplace=True)

# TODO: Participants MUST update this list if they add new features!
FEATURE_COLS = ['MA_Difference'] 
X = df[FEATURE_COLS]
# Note: We use the new binary target column for classification
y = df['Future_Return_Class'] 

train_size = int(len(df) * 0.80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[:train_size])
y_train = y.iloc[:train_size]

df_local_test = df.iloc[train_size:].copy() 
team_name = SUBMISSION_NAME.split('_submission')[0]


# TODO: Tuned the model

print(f"\n--- 2. MODEL TRAINING ({len(X_train_scaled)} samples) ---")

# Random Forest uses an ensemble of trees to make decisions, improving stability.
# Key tuning levers: n_estimators (number of trees) and max_depth (tree complexity).
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100, # Number of trees in the forest
    max_depth=10,     # Max depth of each tree
    criterion='gini'  # Gini impurity for splitting
).fit(X_train_scaled, y_train)

# =========================================================================
#  DO NOT EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

# Get the full path of the current strategy file to save plots correctly
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)

# RUN INDIVIDUAL STOCK ANALYSIS
print("\n--- 3. Running Individual Stock Timing Analysis ---")

X_test_scaled = scaler.transform(df_local_test[FEATURE_COLS])

# Random Forest can use direct prediction (0 or 1) or probability (predict_proba)
# We use direct prediction (0 or 1) here for simplicity.
predicted_class = model.predict(X_test_scaled) 

# The signal is the direct predicted class (1 for up, 0 for down/flat)
df_local_test['Predicted_Return'] = predicted_class # Store the class here for reference
df_local_test['Signal'] = predicted_class

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    # Call the fixed backtesting engine function
    run_single_stock_analysis(df_ticker_data, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH)

# SAVE THE FINAL MODEL
joblib.dump(model, SUBMISSION_NAME)
print(f"\nSUBMISSION READY: Model saved as {SUBMISSION_NAME}")