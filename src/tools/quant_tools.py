import numpy as np
import yfinance as yf
import tensorflow as tf
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler
from langchain_core.tools import tool

# =========================
# CONFIG
# =========================
LOOKBACK = 60
MIN_REQUIRED = 20

# Absolute, Streamlit-safe model path (models folder)
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "lstm_model.h5")
)

# =========================
# DATA FETCHER
# =========================
def _get_yfinance_data(ticker: str, max_retries: int = 3):
    ticker = ticker.strip().upper()

    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers=ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="column",
                progress=False,
            )

            if isinstance(data, pd.DataFrame) and not data.empty:
                if "Close" in data.columns:
                    return data[["Close"]]
                elif isinstance(data.columns, pd.MultiIndex):
                    return data.xs("Close", axis=1, level=0)

        except Exception:
            time.sleep(1 + attempt)

    # Fallback method
    try:
        hist = yf.Ticker(ticker).history(period="1y", auto_adjust=True)
        if not hist.empty and "Close" in hist:
            return hist[["Close"]]
    except Exception:
        pass

    return pd.DataFrame()

# =========================
# MAIN TOOL
# =========================
@tool
def get_technical_signal(ticker: str) -> dict:
    """
    Generates a quantitative trading signal using an LSTM-based deep learning model.
    
    The function:
    - Fetches historical stock price data using yfinance
    - Preprocesses data with MinMax scaling
    - Uses a pre-trained LSTM model to predict the next-day price
    - Returns a bullish or bearish signal with expected percentage change
    
    If data is unavailable, insufficient, or the model fails,
    the function returns a clear error message instead of crashing.
    """
    ticker = ticker.strip().upper()

    # ---- Model Check ----
    if not os.path.exists(MODEL_PATH):
        return {
            "error": "LSTM model not found",
            "summary": f"Quant unavailable: model missing at {MODEL_PATH}",
        }

    # ---- Fetch Data ----
    data = _get_yfinance_data(ticker)

    if data.empty:
        return {
            "error": "No market data from yfinance",
            "summary": "Quant unavailable: no price data.",
        }

    prices = data["Close"].values.flatten()

    if len(prices) < MIN_REQUIRED:
        return {
            "error": f"Only {len(prices)} data points available",
            "summary": "Quant unavailable: insufficient history (<20 days).",
        }

    effective_lookback = min(LOOKBACK, len(prices))

    try:
        scaler = MinMaxScaler()
        scaler.fit(prices.reshape(-1, 1))

        recent_prices = prices[-effective_lookback:].reshape(-1, 1)
        scaled_input = scaler.transform(recent_prices)

        input_data = scaled_input.reshape(1, effective_lookback, 1)

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        prediction_scaled = model.predict(input_data, verbose=0)

        predicted_price = float(
            scaler.inverse_transform(prediction_scaled)[0][0]
        )
        current_price = float(prices[-1])

        percent_change = (
            (predicted_price - current_price) / current_price
        ) * 100

        signal = "BULLISH" if predicted_price > current_price else "BEARISH"

        summary = (
            f"* **Deep Learning Signal:** {signal}\n"
            f"* **Current Price:** ${current_price:.2f}\n"
            f"* **Predicted (next day):** ${predicted_price:.2f}\n"
            f"* **Expected Move:** {percent_change:.2f}%"
        )

        return {
            "error": None,
            "signal": signal,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "percent_change": percent_change,
            "summary": summary,
        }

    except Exception as e:
        return {
            "error": str(e),
            "summary": "Quant unavailable: model inference error.",
        }
