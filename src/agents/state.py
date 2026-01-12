from typing import TypedDict, Dict, Any, Optional


class AgentState(TypedDict, total=False):
    ticker: str

    # Data
    quant: Dict[str, Any]
    sentiment: Dict[str, Any]
    market_data: Dict[str, Any]
    news_data: Any
    rag_context: str

    # Agent outputs
    bull_memo: str
    bear_memo: str

    # Final decision
    final_decision: str
    final_memo: str