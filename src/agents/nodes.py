import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.state import AgentState
from src.tools.quant_tools import get_technical_signal
from src.tools.finance_tools import get_financial_metrics
from src.tools.sentiment_tools import get_news_sentiment
from src.rag.retriever import retrieve_company_documents

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.05
)

# =========================
# RESEARCHER AGENT (WITH RAG)
# =========================
def researcher_node(state: AgentState):
    ticker = state["ticker"]

    # News
    try:
        search = TavilySearchResults(max_results=4)
        news = search.invoke(f"{ticker} earnings risks outlook")
    except Exception:
        news = []

    # Tools
    fundamentals = get_financial_metrics.invoke(ticker)
    quant = get_technical_signal.invoke(ticker)
    sentiment = get_news_sentiment(ticker)

    # 🔥 RAG retrieval
    rag_context = retrieve_company_documents.invoke(
        f"{ticker} long term risks, growth drivers, strategy"
    )

    return {
        "ticker": ticker,
        "market_data": fundamentals,
        "news_data": news,
        "quant_data": quant.get("summary"),
        "quant_raw": quant,
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "rag_context": rag_context
    }

# =========================
# BULL AGENT
# =========================
def bull_node(state: AgentState):
    prompt = f"""
You are a bullish equity analyst.

Write EXACTLY 4 bullet points.
No numbers.

Use:
- Company fundamentals
- Retrieved document context
- Strategic positioning

Document context:
{state['rag_context']}
"""
    return {"bull_memo": llm.invoke(prompt).content.strip()}

# =========================
# BEAR AGENT
# =========================
def bear_node(state: AgentState):
    prompt = f"""
You are a bearish equity analyst.

Write EXACTLY 4 bullet points.
No numbers.

Focus on:
- Risks from documents
- Competitive threats
- Execution challenges

Document context:
{state['rag_context']}
"""
    return {"bear_memo": llm.invoke(prompt).content.strip()}

# =========================
# PM AGENT
# =========================
def pm_node(state: AgentState):
    prompt = f"""
You are a portfolio manager.

Make a BUY / SELL / HOLD decision.

Bull:
{state['bull_memo']}

Bear:
{state['bear_memo']}

Quant: {state['quant_data']}
Sentiment: {state['sentiment_label']}
"""
    return {"final_report": llm.invoke(prompt).content.strip()}
