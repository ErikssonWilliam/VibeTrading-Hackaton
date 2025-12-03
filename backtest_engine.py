import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_max_drawdown(equity_series):
    """Calculates the Maximum Drawdown (MDD) of an equity curve."""
    if equity_series.empty:
        return 0.0
    cumulative_max = equity_series.cummax()
    drawdown = (equity_series - cumulative_max) / cumulative_max
    return drawdown.min() * 100

def run_single_stock_analysis(df_ticker: pd.DataFrame, ticker: str, initial_capital: float, team_name: str, strategy_file_path: str):
    """
    Runs a transaction-based backtest for a single stock, calculates key metrics, 
    and generates a detailed plot showing the bot's timing, including the Alpha Spread.
    
    This function is FIXED and should NOT be edited by participants.
    """
    print(f"\n--- Analyzing {ticker} (Starting with ${initial_capital:,.2f}) ---")
    
    df_prices = df_ticker[['Close']].copy()
    # Shifted signal used for trade on day T (Prediction on T-1 for action on T)
    signal_series = df_ticker['Signal'].shift(1).fillna(0) 

    # Initialize tracking variables
    cash_history = pd.Series(index=df_prices.index, dtype=float)
    portfolio_equity = pd.Series(index=df_prices.index, dtype=float)
    shares_held = pd.Series(0.0, index=df_prices.index)
    
    current_cash = initial_capital
    
    # --- TRANSACTION-BASED SIMULATION ---
    for i, date in enumerate(df_prices.index):
        
        # A. LIQUIDATION AND REBALANCING (Day 2 onwards)
        if i > 0:
            prev_date = df_prices.index[i - 1]
            prev_shares = shares_held.loc[prev_date]
            
            # Value of yesterday's holdings at today's price
            liquidation_value = prev_shares * df_prices.loc[date, 'Close']
            
            # Total capital available for trading
            current_cash = cash_history.loc[prev_date] + liquidation_value
            shares_held.loc[date] = 0.0 # Reset shares after liquidation
        
        # B. TRADING (Allocate based on today's signal)
        signal = signal_series.loc[date]
        
        if signal == 1:
            # Invest all available cash (EWP for single stock)
            investment_amount = current_cash
            price = df_prices.loc[date, 'Close']
            
            if not pd.isna(price) and price > 0:
                shares = investment_amount / price
                shares_held.loc[date] = shares
                current_cash -= investment_amount # Cash balance is now near zero
        
        # C. VALUE PORTFOLIO
        stock_value = shares_held.loc[date] * df_prices.loc[date, 'Close']
        portfolio_value = stock_value + current_cash
        
        # Record history
        cash_history.loc[date] = current_cash
        portfolio_equity.loc[date] = portfolio_value

    # --- ALPHA CALCULATIONS ---
    initial_price = df_prices['Close'].iloc[0]
    buy_and_hold_equity = initial_capital * (df_prices['Close'] / initial_price)
    alpha_spread = portfolio_equity - buy_and_hold_equity # <-- ALPHA SPREAD

    # --- METRICS ---
    daily_returns = portfolio_equity.pct_change().dropna()
    buy_and_hold_return = (df_prices['Close'].iloc[-1] / df_prices['Close'].iloc[0]) - 1.0
    strategy_return = (portfolio_equity.iloc[-1] / initial_capital) - 1.0

    DAYS_IN_YEAR = 252
    mean_return = daily_returns.mean() * DAYS_IN_YEAR
    std_dev_return = daily_returns.std() * np.sqrt(DAYS_IN_YEAR)
    sharpe_ratio = mean_return / std_dev_return if std_dev_return != 0 else 0
    mdd = calculate_max_drawdown(portfolio_equity)

    print(f"  Strategy Return: {strategy_return * 100:.2f}%")
    print(f"  Buy & Hold Return: {buy_and_hold_return * 100:.2f}%")
    print(f"  Annualized Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"  Max Drawdown: {mdd:.2f}%")

    # --- PLOTTING ---
    # Create a figure with two subplots stacked vertically (2 rows, 1 column)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # --- TOP PLOT (Price, Signals, Equity Comparison) ---

    # AXIS 1 (Left): Stock Price (Primary)
    ax1.set_ylabel('Stock Price ($)', color='tab:blue')
    ax1.plot(df_prices.index, df_prices['Close'], color='tab:blue', label=f'{ticker} Price') 
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # AXIS 2 (Right): Equity Curves
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Equity Value ($)', color='black') 
    ax2.plot(portfolio_equity.index, portfolio_equity, color='tab:red', label='Strategy Equity')
    ax2.plot(buy_and_hold_equity.index, buy_and_hold_equity, color='black', linestyle='--', label='Buy & Hold Equity')
    ax2.tick_params(axis='y', labelcolor='black')

    # Plotting Buy Signals
    buy_dates = signal_series[signal_series == 1].index
    buy_prices = df_prices.loc[buy_dates, 'Close']
    ax1.scatter(buy_dates, buy_prices, marker='^', color='green', label='Buy Signal', alpha=0.8, s=50)

    # Legend for Top Plot
    lines, labels = ax1.get_legend_handles_labels() 
    lines2, labels2 = ax2.get_legend_handles_labels() 
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    ax1.set_title(f'Timing Analysis: {ticker} Performance vs. Benchmark ({team_name})')


    # --- BOTTOM PLOT (Alpha Spread Curve) ---

    ax3.plot(alpha_spread.index, alpha_spread, color='purple', label='Alpha Spread')
    ax3.axhline(0, color='gray', linestyle='-', linewidth=1) # The zero line is critical
    ax3.set_ylabel('Alpha Spread ($)')
    ax3.set_xlabel('Date')
    ax3.grid(True, axis='y', linestyle='--')
    ax3.legend(loc='upper left')

    fig.tight_layout()

    # --- SAVE PLOT ---
    output_base_dir = os.path.dirname(strategy_file_path)
    output_plot_dir = os.path.join(output_base_dir, 'backtest_plots')
    os.makedirs(output_plot_dir, exist_ok=True)

    PLOT_FILENAME = f"analysis_{ticker}_{team_name}.png"
    FULL_PATH = os.path.join(output_plot_dir, PLOT_FILENAME)
    plt.savefig(FULL_PATH)
    plt.close()
    print(f"Plot saved successfully to: {FULL_PATH}")
    
    return strategy_return, buy_and_hold_return, sharpe_ratio, mdd