# =========================================================================
# Start of file. Do not edit this section.
# LOGISTIC REGRESSION CLASSIFIER STRATEGY IMPLEMENTATION
# =========================================================================
import pandas as pd
import numpy as np
import joblib
# Import the Logistic Regression model for classification
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)

try:
    # Modules are now imported directly from the root/data folder due to the path fix
    from technical_indicators import macd, rsi  # pyright: ignore[reportMissingImports]
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data 
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary modules. Check project structure and imports. Error: {e}")
    exit()

# LOAD TRAINING DATA
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
SUBMISSION_NAME = 'my_team_name_logreg_submission.joblib'
INITIAL_CAPITAL = 10000.0 
# Buy only if the model predicts the probability of an "Up" move (class 1) is above this threshold.
CLASSIFICATION_THRESHOLD = 0.55 
# ---------------------------------------------------

# SECTION A: FEATURE ENGINEERING AND TARGET DEFINITION

# TODO: Implement additional features as needed 
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']

df['RSI'] = rsi(df['Close'], period=14)
df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = macd(df['Close'], fast=12, slow=26, signal=9)



# Create the Target Variable: CONVERT TO BINARY CLASSIFICATION LABEL
# Y = 1 (Up) if the future return is positive, 0 (Down/Flat) otherwise.
df['Future_Return_Class'] = np.where(df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)) > 0, 1, 0)

df.dropna(inplace=True)

#TODO: Participants MUST update this list if they add new features!
FEATURE_COLS = ['MA_Difference'] 
X = df[FEATURE_COLS]
# Note: Future return is a binary target column for classification
y = df['Future_Return_Class'] 

train_size = int(len(df) * 0.80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X.iloc[:train_size])
y_train = y.iloc[:train_size]

df_local_test = df.iloc[train_size:].copy() 
team_name = SUBMISSION_NAME.split('_submission')[0]

#TODO: Tuned the model
print(f"\n---  MODEL TRAINING ({len(X_train_scaled)} samples) ---")

# Logistic Regression is simple and robust. Participants can try different settings.
model = LogisticRegression(
    random_state=42,
    solver='liblinear',
    C=1.0 # Regularization strength
).fit(X_train_scaled, y_train)

# =========================================================================
# STOP OF EDITING. DO NOT CHANGE BELOW THIS LINE.
# =========================================================================

X_test_scaled = scaler.transform(df_local_test[FEATURE_COLS])

# CRITICAL STEP: Get the probability of the positive class (index 1)
predicted_prob_up = model.predict_proba(X_test_scaled)[:, 1]

# Convert probability into the Signal based on the adjustable threshold
df_local_test['Predicted_Return'] = predicted_prob_up # Store the probability here for reference
df_local_test['Signal'] = np.where(predicted_prob_up > CLASSIFICATION_THRESHOLD, 1, 0)

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    # Call the fixed backtesting engine function
    run_single_stock_analysis(df_ticker_data, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH)

# SUBMIT (SAVE) THE FINAL MODEL
joblib.dump(model, SUBMISSION_NAME)
print(f"\nSUBMISSION READY: Model saved as {SUBMISSION_NAME}")