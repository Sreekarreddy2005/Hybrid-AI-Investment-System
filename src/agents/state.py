from typing import TypedDict, Any


class AgentState(TypedDict):
    ticker: str
    market_data: Any
    news_data: Any
    quant_data: str
    quant_raw: dict
    sentiment_label: str
    sentiment_score: float
    rag_context: str
    bull_memo: str
    bear_memo: str
    final_report: str
