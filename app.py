import streamlit as st
from src.graph import app_graph

# FIRST LINE - Page config
st.set_page_config(page_title="🚀 AI Investment Committee", page_icon="📈", layout="wide")

# Compact CSS - Fixed sizing
st.markdown("""
<style>
.metric-card {background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
              color: white; padding: 1rem; border-radius: 12px; text-align: center; 
              height: 100px; display: flex; flex-direction: column; justify-content: center;}
.decision-badge {background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                 color: white; padding: 1.5rem; border-radius: 20px; text-align: center; margin: 1rem 0;}
.decision-table {background: rgba(255,255,255,0.1); border-radius: 15px; overflow: hidden;}
.decision-table th {background: rgba(255,255,255,0.2); padding: 1rem; text-align: left;}
.decision-table td {padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);}
.bull-card {background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); padding: 1.5rem; border-radius: 15px; color: white;}
.bear-card {background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 1.5rem; border-radius: 15px; color: white;}
</style>
""", unsafe_allow_html=True)

def safe_float(data_dict, key, default=0.0):
    import numpy as np
    value = data_dict.get(key, default)
    if hasattr(value, "item"): return float(value.item())
    if isinstance(value, (np.ndarray, list)): return float(value[0]) if value else default
    return float(value) if value is not None else default

# Header
st.markdown("# 🤖 **Hybrid AI Investment Committee**")
st.markdown("*LSTM + News Sentiment + SEC Analysis → Instant Theses*")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 **Quick Analysis**")
    ticker = st.text_input("**Ticker**", value="TSLA", placeholder="NSE:RELIANCE, NYSE:TSLA").upper()
    if st.button("⚡ **ANALYZE**", use_container_width=True):
        st.rerun()

if ticker:
    with st.spinner(f"🤖 Processing **{ticker}**..."):
        result = app_graph.invoke({"ticker": ticker})
    
    # Header
    st.markdown(f"## 📈 **{ticker}** Investment Thesis")
    
    # Compact Analyst Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="bull-card"><h4>🐂 Bull Case</h4></div>', unsafe_allow_html=True)
        st.markdown(result.get("bull_memo", ""))
    with col2:
        st.markdown('<div class="bear-card"><h4>🐻 Bear Case</h4></div>', unsafe_allow_html=True)
        st.markdown(result.get("bear_memo", ""))
    
    # Compact Metrics (Fixed Size)
    st.markdown("### 📊 **Live Metrics**")
    quant, sentiment = result.get("quant", {}), result.get("sentiment", {})
    metrics = quant.get("metrics", {})
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><strong>${safe_float(metrics, "last_price"):.0f}</strong><br>💰 Price</div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><strong>{safe_float(metrics, "returns_30d"):.1f}%</strong><br>📈 30D Return</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><strong>{safe_float(metrics, "volatility"):.1f}</strong><br>⚡ Volatility</div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><strong>{quant.get("signal")}</strong><br>🎯 LSTM</div>', unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    with c5: st.markdown(f'<div class="metric-card"><strong>{sentiment.get("sentiment")}</strong><br>📰 Sentiment</div>', unsafe_allow_html=True)
    with c6: st.markdown(f'<div class="metric-card"><strong>{sentiment.get("newscount")}</strong><br>🗞️ News</div>', unsafe_allow_html=True)
    
    # Clean Decision Section (No Code)
    st.markdown("### 🎯 **Investment Decision**")
    decision = result.get("final_decision", "HOLD")
    
    if decision == "BUY":
        st.markdown('<div class="decision-badge"><h2>✅ **STRONG BUY**</h2></div>', unsafe_allow_html=True)
    elif decision == "SELL":
        st.markdown('<div class="decision-badge" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);"><h2>❌ **STRONG SELL**</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="decision-badge" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);"><h2>⏸️ **HOLD**</h2></div>', unsafe_allow_html=True)
    
    # Clean Table (Pure HTML)
    st.markdown(f'''
    <div class="decision-table">
        <table>
            <tr><th>Metric</th><th>{ticker}</th></tr>
            <tr><td>💰 Latest Price</td><td><strong>${safe_float(metrics, "last_price"):.2f}</strong></td></tr>
            <tr><td>📈 30-Day Return</td><td><strong>{safe_float(metrics, "returns_30d"):.2f}%</strong></td></tr>
            <tr><td>⚡ Volatility</td><td><strong>{safe_float(metrics, "volatility"):.2f}</strong></td></tr>
            <tr><td>🎯 LSTM Signal</td><td><strong style="color: #10b981;">{quant.get("signal")}</strong></td></tr>
            <tr><td>📰 News Sentiment</td><td><strong>{sentiment.get("sentiment")}</strong></td></tr>
        </table>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown(f"**AI Thesis**: {result.get('final_memo', 'Analysis complete')}")
else:
    st.info("👈 Enter ticker (works for **USA + India**: NSE:RELIANCE, NYSE:TSLA)")
