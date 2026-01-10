# src/agents/state.py
from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict, total=False):
    ticker: str

    # Raw data
    market_data: Dict[str, Any]          # fundamentals from yfinance
    news_data: List[Dict[str, Any]]      # Tavily news list
    quant_data: str                      # quant summary string
    quant_raw: Dict[str, Any]            # numeric + error fields

    # Sentiment
    sentiment_score: float
    sentiment_label: str

    # Agent outputs
    bull_memo: str
    bear_memo: str
    final_report: str
