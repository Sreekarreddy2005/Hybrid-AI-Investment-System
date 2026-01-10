# src/tools/sentiment_tools.py
import os
import requests

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def _map_finnhub_score(score: float) -> str:
    if score > 0.15:
        return "Bullish"
    elif score < -0.15:
        return "Bearish"
    return "Neutral"

def get_news_sentiment(ticker: str) -> dict:
    """
    Uses Finnhub's news sentiment API to get a sentiment score/label.
    Falls back to neutral if API fails.
    """
    ticker = ticker.strip().upper()
    if not FINNHUB_API_KEY:
        return {"score": 0.0, "label": "Neutral", "source": "Sentiment API disabled (no key)"}

    try:
        url = "https://finnhub.io/api/v1/news-sentiment"
        params = {"symbol": ticker, "token": FINNHUB_API_KEY}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        score = float(data.get("sentiment", 0.0))
        label = _map_finnhub_score(score)

        return {"score": score, "label": label, "source": "Finnhub News Sentiment"}
    except Exception:
        return {"score": 0.0, "label": "Neutral", "source": "Sentiment API error -> Neutral"}
