# 💰✨ VibeTrading Hackathon: Machine Learning Trading Challenge

Welcome to the **VibeTrading Hackathon**! Your mission is to develop a machine learning strategy that correctly captures the market's "vibe" (momentum and reversals) to generate **Alpha** returns greater than the simple Buy and Hold benchmark (Return better than simple holding the Stock).

**Python:** The programming language you'll be using to write your AI. 

**VSCode (or other code editor):** The code editor you'll use to write and edit your Python code.  

**Git:** Version control tool you'll use to download the hackaton code and save your changes.

## Setup Instructions

Choose your operating system for complete setup instructions:

- [**Windows Setup Guide**](Windows_setup.md) - Complete installation guide for Windows
- [**macOS Setup Guide**](Mac_setup.md) - Complete installation guide for Mac
- [**Linux Setup Guide**](Linux_setup.md) - Complete installation guide for Linux

## 🏆 The Challenge: Maximize Alpha and Control Risk

The winner will be the team that creates the most profitable bot that successfully manages risk.

* **Primary Metric:** **Alpha** (The bot's return - Buy and Hold Return). (Higher is better).

---

## 📂 Project Structure

The project is modular. 

| Folder / File | Purpose | Shall be edited? |
| :--- | :--- | :--- |
| `backtest_engine.py` | The fixed simulation environment, metrics, and plotting logic. | **NO** |
| `data/load_data.py` | Loads the historical stock data used for training and testing. | **NO** |
| `"bot_name/` | **Your primary workspace.** Contains the starting templates for each bot. | **YES** |
| `strategies/<bot_name>/` | Dedicated folders for each strategy, including output plots. | **YES** |
| `README.md` | This guide. | **NO** |

---

Note that if any participants modify the files which shall not be edited, no competitive advantage will be gained.

---
## 🐍 The 5 Starter Bots (The VibeTrading Team)

You have five pre-built Machine Learning bots, each using a distinct core algorithm to beat the market.

| Bot Name | Technique | Approach | Key $\text{sklearn}$ Model |
| :--- | :--- | :--- | :--- |
| **Hyperplane Hippie** 🔄 | Multi-Layer Perceptron | **Regression** (Predicts Return Magnitude) | `MLPRegressor` |
| **Lord of the Bins** 🗑️ | Logistic Regression | **Classification** (Predicts Up/Down Probability) | `LogisticRegression` |
| **The Baguette Booster** 🥖 | Random Forest | **Classification** (Ensemble Voting) | `RandomForestClassifier` |
| **KNN Cash Cow** 🐄 | K-Nearest Neighbors | **Classification** (Proximity/Historical Match) | `KNeighborsClassifier` |
| **"My cool bot"** 📐 | Up to the user | **Classification/Regression** | (To be implemented) |

---

## 🛠️ How to Improve the Bots (The Three Levers)

To generate Alpha, you must modify the strategy templates using the three primary levers listed below.

### 1. Feature Engineering (The Alpha Source)
* **Location:** Look for the section `🎯 SECTION A: FEATURE ENGINEERING`.
* **Goal:** Create new, predictive input features ($\text{X}$) that the $\text{ML}$ model can learn from. The current bots only use a simple Moving Average Difference.
* **Ideas:**
    * **Momentum:** Relative Strength Index ($\text{RSI}$), $\text{MACD}$ components.
    * **Volatility:** Average True Range ($\text{ATR}$), volume-adjusted standard deviations.
    * **Market Context:** Compare the stock's performance to the overall $\text{SP 500}$ market.
* **Critical Step:** If you add a feature, **you MUST update the `FEATURE_COLS` list** inside the strategy file.

### 2. Prediction Horizon & Target Tuning
* **Location:** Look for the `N_DAYS_PREDICT` variable in the `--- CONFIGURATION ---` section.
* **Goal:** Adjust how far into the future the model tries to predict (e.g., short-term flips or long-term trends).
* **Classification Bots Only:** Tune the **confidence threshold** (e.g., in `Lord of the Bins`) to only buy if the model is highly confident (e.g., $55\%$ probability of "Up" or higher).

### 3. Model Hyperparameter Tuning
* **Location:** Look for the section `🎯 SECTION B: TRAIN MODEL`.
* **Goal:** Adjust the core settings of the $\text{ML}$ algorithm to fit the data better and reduce overfitting.
* **Examples:**
    * **MLP:** Change `hidden_layer_sizes`, `max_iter`.
    * **Random Forest:** Change `n_estimators` (number of trees), `max_depth`.
    * **KNN:** Change `n_neighbors` (the 'K' value).

---

## ⚙️ How to Run Your Strategy

1.  Open the corresponding Python file (e.g., `strategies/knn_cash_cow/knn_cash_cow.py`).
2.  Make your changes to **Sections A and B**.
3.  Run the file from your terminal (from the project root):
    ```bash
    python strategies/knn_cash_cow/knn_strategy_template.py
    ```
4.  The script will print metrics to the console and save the visual analysis plot.

### 🖼️ Interpreting the Alpha Plot

The output plot is now split into two panels for clearer analysis:

* **Top Panel:** Shows the Stock Price, your **Strategy Equity** (Red), and the **Buy and Hold Equity** (Black Dashed Line). Green triangles show your bot's **Buy Signals**.
* **Bottom Panel (The Alpha Spread):** Shows the difference between your strategy's equity and Buy and Hold ($\text{Strategy} - \text{B and H}$).
    * If the purple line is **above the zero axis**, your bot is currently **outperforming** the market.
    * If the purple line is **trending upwards**, your bot is successfully generating Alpha during that period.
 
## Tips and Tricks
 
Read here for more tips and tricks towards a winning strategy: - [**Tips and Tricks**](Tips_N_Tricks.md)
