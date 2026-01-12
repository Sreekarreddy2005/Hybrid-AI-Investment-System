import numpy as np
import tensorflow as tf
import yfinance as yf

MODEL_PATH = "models/lstm_model.h5"
LOOKBACK = 60

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception:
    model = None


def get_technical_signal(ticker: str) -> dict:
    ticker = ticker.upper().strip()

    if model is None:
        return {"status": "error", "message": "LSTM model not loaded"}

    try:
        data = yf.download(ticker, period="6mo", interval="1d", progress=False)

        if len(data) < LOOKBACK:
            raise ValueError("Insufficient historical data")

        close = data["Close"].values
        last_price = float(close[-1])
        avg_60 = float(close[-LOOKBACK:].mean())
        volatility = float(close[-LOOKBACK:].std())
        returns_30d = ((close[-1] / close[-30]) - 1) * 100

        # Normalize for LSTM
        mean = close[-LOOKBACK:].mean()
        std = close[-LOOKBACK:].std() + 1e-8
        normalized = (close[-LOOKBACK:] - mean) / std
        X = normalized.reshape(1, LOOKBACK, 1)

        prediction = float(model.predict(X, verbose=0)[0][0])
        signal = "Bullish" if prediction > 0 else "Bearish"

        return {
            "status": "ok",
            "signal": signal,
            "summary": f"LSTM predicts a {signal.lower()} outlook based on recent price action.",
            "metrics": {
                "last_price": last_price,
                "avg_60": avg_60,
                "volatility": volatility,
                "returns_30d": returns_30d,
                "lstm_output": prediction,
            },
            "price_data": data["Close"]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
