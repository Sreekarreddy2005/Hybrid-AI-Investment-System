from typing import TypedDict

class AgentState(TypedDict):
    ticker: str
    market_data: str      # Raw financial data (text)
    quant_data: str       # LSTM prediction (text)
    news_data: str        # News headlines (text)
    bull_memo: str        # Output from Bull Agent
    bear_memo: str        # Output from Bear Agent
    final_report: str     # Output from PM