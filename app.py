import streamlit as st
from dotenv import load_dotenv
from src.graph import app_graph

load_dotenv()

st.set_page_config(page_title="AI Hedge Fund", layout="wide")
st.title("🤖 Hybrid AI Investment Committee")
st.markdown("### Architecture: LangGraph (Agents) + TensorFlow (LSTM) + Tavily (RAG)")

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    ticker = st.text_input("Enter Stock Ticker:", value="AAPL").upper()
    run_btn = st.button("Generate Investment Thesis", type="primary")

if run_btn:
    if not ticker:
        st.warning("Please enter a valid ticker.")
    else:
        with st.spinner(f"Agents are analyzing {ticker}..."):
            try:
                # Run Graph
                result = app_graph.invoke({"ticker": ticker})
                
                # --- FIX: Clean Up Text (Escape $ signs) ---
                # This prevents Streamlit from deleting spaces and using math font
                bull_clean = result['bull_memo'].replace("$", "\\$")
                bear_clean = result['bear_memo'].replace("$", "\\$")
                final_report_clean = result['final_report'].replace("$", "\\$")
                
                # --- Bull & Bear Section ---
                col1, col2 = st.columns(2)
                with col1:
                    st.success("🐂 Bull Case")
                    st.markdown(bull_clean) # Use cleaned text
                with col2:
                    st.error("🐻 Bear Case")
                    st.markdown(bear_clean) # Use cleaned text
                    
                st.divider()
                
                # --- Quant Section ---
                st.subheader("🧠 Deep Learning Quant Signal")
                st.info(result['quant_data'])
                
                st.divider()
                
                # --- Final Decision Section ---
                st.header("👔 Final Decision")
                st.markdown(final_report_clean) # Use cleaned text
                
            except Exception as e:
                st.error(f"An error occurred: {e}")