import yfinance as yf
from langchain_core.tools import tool

@tool
def get_financial_metrics(ticker: str):
    """Retrieves key valuation metrics."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "pe_ratio": info.get("forwardPE", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "recommendation": info.get("recommendationKey", "none")
        }
    except Exception:
        return "Failed to fetch financial metrics."