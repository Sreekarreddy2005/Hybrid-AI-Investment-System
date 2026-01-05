import os
from dotenv import load_dotenv

# 1. LOAD KEYS
load_dotenv()

from langchain_core.messages import SystemMessage
# CHANGED: Import Groq instead of OpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from src.agents.state import AgentState
from src.tools.finance_tools import get_financial_metrics
from src.tools.quant_tools import get_technical_signal

# 2. Check for Groq Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found. Please check your .env file.")

# 3. INITIALIZE GROQ LLM
# We use 'llama3-70b-8192' which is powerful and free on Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # <--- CHANGED THIS
    temperature=0.2
)

def researcher_node(state: AgentState):
    """Runs Tools: Tavily (News) + YFinance (Data) + LSTM (Quant)"""
    ticker = state['ticker']
    
    # 1. News
    search = TavilySearchResults(max_results=3)
    try:
        news_results = search.invoke(f"{ticker} stock news analysis")
    except Exception as e:
        news_results = f"Error fetching news: {str(e)}"
    
    # 2. Financials
    try:
        financial_data = get_financial_metrics.invoke(ticker)
    except Exception as e:
        financial_data = "Error fetching financials."
    
    # 3. Quant (Deep Learning)
    # We add a try/except here too just in case the LSTM file has issues
    try:
        quant_signal = get_technical_signal.invoke(ticker)
    except Exception as e:
        quant_signal = "Quant Model Error (Check logs)"
    
    return {
        "news_data": str(news_results),
        "market_data": str(financial_data),
        "quant_data": str(quant_signal)
    }

def bull_node(state: AgentState):
    prompt = f"""
    Role: Bullish Analyst. 
    Task: Write a convincing argument to BUY {state['ticker']}.
    
    Data:
    - Financials: {state['market_data']}
    - Deep Learning Signal: {state['quant_data']}
    - News: {state['news_data']}
    
    Focus on growth, upside potential, and technical strength. Be concise (3 bullet points).
    """
    # Groq works best with simple string prompts
    res = llm.invoke(prompt)
    return {"bull_memo": res.content}

def bear_node(state: AgentState):
    prompt = f"""
    Role: Bearish Short-Seller. 
    Task: Write a convincing argument to SELL {state['ticker']}.
    
    Data:
    - Financials: {state['market_data']}
    - Deep Learning Signal: {state['quant_data']}
    - News: {state['news_data']}
    
    Focus on risks, overvaluation, and negative news. Be concise (3 bullet points).
    """
    res = llm.invoke(prompt)
    return {"bear_memo": res.content}

def pm_node(state: AgentState):
    prompt = f"""
    Role: Portfolio Manager.
    Task: Make a final Buy/Sell/Hold decision.
    
    Inputs:
    - Bull Memo: {state['bull_memo']}
    - Bear Memo: {state['bear_memo']}
    - Quant Signal: {state['quant_data']}
    
    Output Format:
    ## Decision: [BUY / SELL / HOLD]
    ### Reasoning:
    [Summary]
    ### Confidence Score: [0-100]%
    """
    res = llm.invoke(prompt)
    return {"final_report": res.content}