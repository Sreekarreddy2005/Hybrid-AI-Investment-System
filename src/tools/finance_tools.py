# src/tools/finance_tools.py
import yfinance as yf
from langchain_core.tools import tool

@tool
def get_financial_metrics(ticker: str) -> dict:
    """Retrieve key valuation & fundamental metrics."""
    ticker = ticker.strip().upper()
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "error": None,
            "pe_ratio": info.get("forwardPE"),
            "trailing_pe": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "recommendation": info.get("recommendationKey", "none"),
            "sector": info.get("sector"),
            "short_name": info.get("shortName"),
        }
    except Exception as e:
        return {
            "error": f"Failed to fetch financial metrics: {str(e)}",
        }
