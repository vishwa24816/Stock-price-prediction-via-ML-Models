# Stock Price Prediction: Linear Regression vs Random Forest vs XGBoost vs LSTM

A 5-cell Jupyter notebook that downloads daily stock data from Yahoo Finance, trains four machine learning models on 2000-2019 data, and compares their predictions against actual prices for 2020-2025.

## What it does

1. Downloads daily (1d) OHLC data for a chosen ticker using `yfinance`
2. Engineers return-based and trend-based features
3. Trains four models: Linear Regression, Random Forest, XGBoost and LSTM
4. Plots each model's predictions against actual prices for 2020-2025
5. Produces one consolidated graph (actual + 4 predictions) and a metrics table showing which model followed the actual price most closely

## Requirements

- Python 3.9+
- Internet access (for the Yahoo Finance download and `pip install`)
- Packages (installed automatically by Cell 1):
  - `yfinance`
  - `xgboost`
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `numpy`, `pandas`

Google Colab works well out of the box. A GPU is not required, but speeds up the LSTM.

## How to run

1. Open a new Jupyter or Colab notebook
2. Paste each of the 5 cells into its own cell, in order
3. (Optional) Change `TICKER = "AAPL"` in Cell 1 to any Yahoo Finance symbol (e.g. `"MSFT"`, `"TSLA"`, `"RELIANCE.NS"`, `"^NSEI"`)
4. Run the cells top to bottom. Each cell depends on variables from the previous ones, so do not skip or reorder them.

## Cell-by-cell breakdown

### Cell 1: Setup and data download
- Installs dependencies and imports libraries
- Downloads daily data from 2000-01-01 to 2025-12-31 with `auto_adjust=True` (prices adjusted for splits and dividends)
- Keeps only the `Close` column and plots the full price history

### Cell 2: Feature engineering and train/test split
Creates 19 features per day:

| Group | Features | Count |
|---|---|---|
| Lagged returns | Log returns of the last 10 days (`lag1`..`lag10`) | 10 |
| Rolling mean of returns | 5, 10, 20-day windows | 3 |
| Rolling volatility | Std of returns over 5, 10, 20-day windows | 3 |
| Trend ratios | `Close / SMA - 1` for 10, 20, 50-day SMAs | 3 |

- **Target:** next-day log return (`target`)
- **Reference price:** next-day close (`NextClose`), used as the actual value for comparison
- **Split:** rows before 2020-01-01 are training data; everything from 2020 onward is the test set
- Features are standardized with a `StandardScaler` fit on the training set only, which avoids data leakage

### Cell 3: Classical ML models
Trains and plots three models:

| Model | Key settings |
|---|---|
| Linear Regression | Defaults |
| Random Forest | 300 trees, max depth 8, min leaf size 5 |
| XGBoost | 400 trees, learning rate 0.03, max depth 4, subsample 0.8, colsample 0.8 |

Each model predicts next-day return; the prediction is converted to a price (see Methodology). One graph per model is displayed.

### Cell 4: LSTM
- Builds sequences of the previous 30 trading days of features as input
- Architecture: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)
- Optimizer: Adam, loss: MSE, up to 50 epochs, batch size 64
- Early stopping (patience 5) on a validation split taken from the last 10% of the training data (no shuffling, so time order is respected)
- Predictions for the test period are converted to prices and plotted

### Cell 5: Consolidated graph and evaluation
- Plots the actual price (black, thick) with all four model predictions on one chart
- Computes metrics for every model and ranks them by RMSE
- Prints the model closest to the actual price and the model with the best direction accuracy

## Methodology

### Why predict returns instead of prices?
Tree-based models (Random Forest, XGBoost) cannot output values outside the range they saw in training. If a stock trades far above its 2000-2019 maximum during 2020-2025, a model predicting raw prices would flatline at that ceiling. Returns are roughly stationary, so models trained on them generalize across price levels.

### Converting predictions back to price
```
predicted_price(t+1) = actual_close(t) * exp(predicted_log_return(t+1))
```

This is a **one-step-ahead** forecast: each day's prediction is anchored to the previous day's *actual* close.

## Evaluation metrics

| Metric | Meaning | Better |
|---|---|---|
| RMSE | Root mean squared error in price units; penalizes large misses | Lower |
| MAE | Average absolute error in price units | Lower |
| MAPE % | Average percentage error | Lower |
| Direction Acc % | How often the model correctly predicted whether the price would go up or down vs. the previous close | Higher (50% = coin flip) |

The results table is sorted by RMSE, so the top row is the model that followed the actual price most closely.

## Output

- 1 price history plot (Cell 1)
- 4 individual model-vs-actual plots (Cells 3 and 4)
- 1 consolidated plot with all predictions (Cell 5)
- 1 metrics table and a printed summary of the best models

## Customization

- **Different stock:** change `TICKER` in Cell 1
- **Different split date:** change `"2020-01-01"` in Cells 2 and 4 (keep them consistent)
- **More or fewer features:** edit the loops in Cell 2; `FEATS` updates automatically
- **LSTM window:** change `WIN` in Cell 4
- **Tuning:** adjust hyperparameters in the `models` dictionary (Cell 3) or the LSTM layers (Cell 4)

## Limitations and caveats

- **Not a long-range forecast.** Predictions are next-day, using real prices up to the previous day. The notebook does not generate a free-running 5-year forecast, which would drift badly for every model.
- **The lines will hug the actual price.** Because each prediction is anchored to yesterday's actual close, all models look accurate on the chart. Differences show up in the metrics, not visually.
- **Hard baseline.** Daily returns are close to random, so a naive "tomorrow = today" forecast is very hard to beat. Direction accuracy near 50-55% is typical. Treat large gains with suspicion and check for leakage.
- **No transaction costs or trading logic.** These are prediction metrics, not a backtested strategy.
- **Randomness.** LSTM and tree results vary slightly between runs. Seeds are set, but GPU nondeterminism can still cause small differences.
- **Single stock, single split.** Results on one ticker or one time period do not generalize. For more robust conclusions, use walk-forward validation across several tickers.

## Troubleshooting

| Problem | Fix |
|---|---|
| Empty DataFrame from `yfinance` | Check the ticker symbol and your internet connection; try again later (rate limits) |
| `KeyError` or column errors after download | Update `yfinance` (`pip install -U yfinance`) |
| TensorFlow install fails | Use Colab, or install a TensorFlow version that matches your Python version |
| `NameError` in a later cell | Run all previous cells first, in order |
| LSTM is slow | Use a GPU runtime, or reduce `epochs` / LSTM units |

## Disclaimer

This project is for educational purposes only. It is not financial advice, and past model performance does not indicate future results.
