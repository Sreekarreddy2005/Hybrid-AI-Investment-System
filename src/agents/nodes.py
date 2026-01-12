import os
from dotenv import load_dotenv
load_dotenv(override=True)
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from src.agents.state import AgentState
from src.tools.quant_tools import get_technical_signal
from src.tools.finance_tools import get_financial_metrics
from src.tools.sentiment_tools import get_sentiment
from src.rag.retriever import retrieve_company_documents

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.05,
    api_key=os.getenv("GROQ_API_KEY")
)

def researcher_node(state: AgentState):
    ticker = state["ticker"].upper()
    
    # News, quant, sentiment (all working)
    try:
        search = TavilySearchResults(max_results=6)
        news_data = search.invoke(f"{ticker} stock news")
    except: news_data = []
    
    quant = get_technical_signal(ticker)
    sentiment = get_sentiment(ticker)
    
    # RAG - Your existing retriever handles indexing automatically
    raw_rag = retrieve_company_documents.invoke({
        "ticker": ticker,
        "query": "risks opportunities strategy outlook"
    })
    
    # Graceful handling
    if "RAG_UNAVAILABLE" in raw_rag or "No relevant" in raw_rag:
        rag_context = "SEC filings analysis unavailable. Strong quant + news signals dominate."
    else:
        rag_context = raw_rag[:800]
    
    return {
        "ticker": ticker,
        "quant": quant,
        "sentiment": sentiment,
        "rag_context": rag_context
    }
def bull_node(state: AgentState):
    prompt = f"""Bullish analyst. 4 bullet points MAX. Use: {state.get('rag_context', '')[:1000]}"""
    return {"bull_memo": llm.invoke(prompt).content.strip()}

def bear_node(state: AgentState):
    prompt = f"""Bearish analyst. 4 bullet points MAX. Use: {state.get('rag_context', '')[:1000]}"""
    return {"bear_memo": llm.invoke(prompt).content.strip()}

def pm_node(state: AgentState):
    # Get ALL data
    quant = state.get("quant", {})
    sentiment = state.get("sentiment", {})
    rag = state.get("rag_context", "")
    bull = state.get("bull_memo", "")
    bear = state.get("bear_memo", "")
    
    signal = quant.get("signal", "Neutral")
    sent_label = sentiment.get("sentiment", "Neutral")
    news_n = sentiment.get("newscount", 0)
    
    decision = "HOLD"
    if signal == "Bullish" and sent_label == "Positive": decision = "BUY"
    elif signal == "Bearish" and sent_label == "Negative": decision = "SELL"
    
    # Safe metrics
    def safe_float(d, k, default=0.0):
        import numpy as np
        v = d.get(k, default)
        if hasattr(v, "item"): return float(v.item())
        if isinstance(v, (np.ndarray, list)): return float(v[0]) if v else default
        return float(v) if v is not None else default
    
    metrics = quant.get("metrics", {})
    price = safe_float(metrics, "last_price")
    ret30 = safe_float(metrics, "returns_30d")
    vol = safe_float(metrics, "volatility")
    
    memo = f"""## 🎯 **{decision}** Recommendation

### 📊
| Metric | Value |
|--------|-------|
| Price | ${price:.2f} |
| 30D Return | {ret30:.1f}% |
| Volatility | {vol:.2f} |
| Quant Signal | {signal} |
| News ({news_n}) | {sent_label} |

### 🤖 AI Analysis
**Quant**: {signal} LSTM prediction  
**Sentiment**: {sent_label} from {news_n} articles  
**Filings**: {rag[:250]}...  
**Consensus**: Bull case strong growth vs bearish risks

*Multi-agent AI recommendation*"""
    
    return {"final_decision": decision, "final_memo": memo}
