import streamlit as st
from dotenv import load_dotenv
from src.graph import app_graph

load_dotenv()

st.set_page_config(
    page_title="Hybrid AI Investment Committee",
    layout="wide"
)

st.title("🤖 Hybrid AI Investment Committee")
st.caption("LangGraph Agents + LSTM Quant + Sentiment Analysis")

with st.sidebar:
    st.header("Control Panel")
    ticker = st.text_input("Enter Stock Ticker", value="TSLA").upper()
    run_btn = st.button("Generate Investment Thesis", type="primary")

if run_btn:
    if not ticker:
        st.warning("Please enter a valid ticker.")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                result = app_graph.invoke({"ticker": ticker})

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🐂 Bull Case")
                    st.markdown(result["bull_memo"])

                with col2:
                    st.subheader("🐻 Bear Case")
                    st.markdown(result["bear_memo"])

                st.divider()
                st.subheader("🧠 Quant & Sentiment")

                quant_error = result["quant_raw"].get("error")
                if quant_error:
                    st.warning(f"Quant unavailable: {quant_error}")
                else:
                    st.markdown(result["quant_data"])

                st.divider()
                st.subheader("👔 Final Investment Memo")
                st.markdown(result["final_report"])

                st.caption("⚠️ Educational use only. Not financial advice.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")
