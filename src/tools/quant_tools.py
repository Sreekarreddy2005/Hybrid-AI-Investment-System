import numpy as np
import yfinance as yf
import tensorflow as tf
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from langchain_core.tools import tool

# Config
# We use os.getcwd() to find the absolute path safely
MODEL_PATH = os.path.join(os.getcwd(), "models", "lstm_model.h5")
LOOKBACK = 60

@tool
def get_technical_signal(ticker: str):
    """
    Uses the LSTM Deep Learning model to predict trend direction.
    Uses standard yfinance connection (v0.2.40+) to avoid blocking.
    """
    # 1. CLEAN THE TICKER
    ticker = ticker.strip().upper()

    if not os.path.exists(MODEL_PATH):
        return "Error: lstm_model.h5 not found in models/ directory."

    try:
        # 2. LOAD MODEL
        # compile=False prevents version mismatch errors
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # 3. FETCH REAL DATA
        # print(f"DEBUG: Fetching REAL data for '{ticker}'...") # Optional debug
        
        try:
            # Do NOT pass a session object. Let yfinance handle it.
            stock = yf.Ticker(ticker)
            data = stock.history(period="2y")
        except Exception as e:
            return f"Error: Yahoo API Connection Failed. Reason: {str(e)}"

        # 4. VALIDATE DATA
        if len(data) == 0:
            return f"Error: Yahoo Finance returned 0 rows for '{ticker}'. Your IP might be temporarily blocked. Try again later."
        
        if len(data) < LOOKBACK:
            return f"Error: Only found {len(data)} days of data. Need {LOOKBACK}+ days for ML inference."

        # 5. PREPROCESS
        # Handle different yfinance return formats
        if isinstance(data.columns, pd.MultiIndex):
            try:
                prices = data[('Close', ticker)].values
            except KeyError:
                prices = data['Close'].iloc[:, 0].values
        else:
            prices = data['Close'].values

        # Take last 60 days
        prices = prices[-LOOKBACK:].reshape(-1, 1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices)
        input_data = np.reshape(scaled_data, (1, LOOKBACK, 1))
        
        # 6. PREDICT
        prediction_scaled = model.predict(input_data, verbose=0)
        prediction_price = scaler.inverse_transform(prediction_scaled)[0][0]
        current_price = prices[-1][0]
        
        # 7. INTERPRET
        percent_change = ((prediction_price - current_price) / current_price) * 100
        signal = "BULLISH 📈" if prediction_price > current_price else "BEARISH 📉"
        
        # 8. RETURN FORMATTED STRING (Bullet points for nice display)
        return (f"* **Deep Learning Signal:** {signal}\n"
                f"* **Current Price:** ${current_price:.2f}\n"
                f"* **Predicted (24h):** ${prediction_price:.2f}\n"
                f"* **Change Exp:** {percent_change:.2f}%")

    except Exception as e:
        return f"Quant Tool Critical Error: {str(e)}"