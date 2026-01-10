import os
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

from src.agents.state import AgentState
from src.tools.finance_tools import get_financial_metrics
from src.tools.quant_tools import get_technical_signal
from src.tools.sentiment_tools import get_news_sentiment

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")

# 🔒 Low temperature + concise output = stable formatting
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.03,
)

# =========================
# RESEARCHER NODE
# =========================
def researcher_node(state: AgentState):
    ticker = state["ticker"]

    # News (titles only to reduce verbosity)
    try:
        search = TavilySearchResults(max_results=4)
        news_results = search.invoke(
            f"{ticker} stock earnings risks outlook analyst opinion"
        )
        news = [{"title": n.get("title")} for n in news_results]
    except Exception:
        news = []

    fundamentals = get_financial_metrics.invoke(ticker)
    quant = get_technical_signal.invoke(ticker)
    sentiment = get_news_sentiment(ticker)

    return {
        "ticker": ticker,
        "market_data": fundamentals,
        "news_data": news,
        "quant_data": quant.get("summary"),
        "quant_raw": {
            "signal": quant.get("signal"),
            "error": quant.get("error"),
            "current_price": quant.get("current_price"),
            "predicted_price": quant.get("predicted_price"),
            "percent_change": quant.get("percent_change"),
        },
        "sentiment_label": sentiment["label"],
        "sentiment_score": sentiment["score"],
    }

# =========================
# BULL NODE (STRICT FORMAT)
# =========================
def bull_node(state: AgentState):
    prompt = f"""
You are a professional equity analyst.

Write a BULL CASE for {state['ticker']}.

STRICT RULES (MANDATORY):
- EXACTLY 4 bullet points
- Each bullet must be ONE sentence
- DO NOT use numbers, prices, percentages, or tickers
- No line breaks inside bullets
- Professional analyst tone
- No repetition

Context:
Fundamentals: {state['market_data']}
Quant Direction: {state['quant_raw'].get('signal')}
Sentiment: {state['sentiment_label']}
News: {state['news_data']}

FORMAT EXACTLY:

## Bull Case
- Bullet one sentence
- Bullet one sentence
- Bullet one sentence
- Bullet one sentence
"""
    res = llm.invoke(prompt)
    return {"bull_memo": res.content.strip()}

# =========================
# BEAR NODE (STRICT FORMAT)
# =========================
def bear_node(state: AgentState):
    prompt = f"""
You are a professional equity analyst.

Write a BEAR CASE for {state['ticker']}.

STRICT RULES (MANDATORY):
- EXACTLY 4 bullet points
- Each bullet must be ONE sentence
- DO NOT use numbers, prices, percentages, or tickers
- No truncation
- No markdown errors
- Professional risk-focused tone

Context:
Fundamentals: {state['market_data']}
Quant Direction: {state['quant_raw'].get('signal')}
Sentiment: {state['sentiment_label']}
News: {state['news_data']}

FORMAT EXACTLY:

## Bear Case
- Bullet one sentence
- Bullet one sentence
- Bullet one sentence
- Bullet one sentence
"""
    res = llm.invoke(prompt)
    return {"bear_memo": res.content.strip()}

# =========================
# PORTFOLIO MANAGER NODE
# =========================
def pm_node(state: AgentState):
    prompt = f"""
You are a hedge fund portfolio manager.

Make a FINAL investment decision for {state['ticker']}.

STRICT RULES:
- Decision must be BUY, SELL, or HOLD
- Use bullet points only
- No numbers inside sentences
- Conservative, risk-aware tone
- Clear and complete sentences

Inputs:
Bull Case:
{state['bull_memo']}

Bear Case:
{state['bear_memo']}

Quant Direction: {state['quant_raw'].get('signal')}
Sentiment: {state['sentiment_label']}
Fundamentals: {state['market_data']}

FORMAT EXACTLY:

## Decision: BUY / SELL / HOLD

### Reasoning
- Bullet one sentence
- Bullet one sentence
- Bullet one sentence

### Risk & Positioning
- Time horizon
- Position size
- Key risk

### Confidence Score
- XX percent
- One sentence justification
"""
    res = llm.invoke(prompt)
    return {"final_report": res.content.strip()}
