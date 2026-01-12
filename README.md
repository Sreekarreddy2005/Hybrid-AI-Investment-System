# Hybrid-AI-Investment-System
# 🤖 Hybrid AI Investment System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)

*An intelligent multi-agent AI investment analysis platform combining LSTM predictions, sentiment analysis, and RAG-powered SEC filings research*

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Architecture](#-architecture) • [Usage](#-usage)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Multi-Agent Workflow](#-multi-agent-workflow)
- [Example Output](#-example-output)
- [API Keys Required](#-api-keys-required)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#-disclaimer)

## 🎯 Overview

The **Hybrid AI Investment System** is a production-ready multi-agent AI platform that automates investment analysis by combining:

- **🧠 LSTM Neural Network** for price prediction
- **📰 Real-time News Sentiment Analysis** using Tavily Search
- **📄 RAG (Retrieval-Augmented Generation)** for SEC 10-K filing analysis
- **🤖 Multi-Agent Collaboration** with LangGraph orchestration
- **💡 LLM-Powered Investment Theses** via Groq's Llama 3.3 70B

This system simulates an AI-powered investment committee where specialized agents (Bull Analyst, Bear Analyst, Portfolio Manager) collaborate to produce comprehensive investment recommendations.

### 🎓 Educational & Research Purpose

This project demonstrates advanced AI/ML techniques in quantitative finance and is intended for **educational purposes only**. It showcases:
- Multi-agent AI systems with LangGraph
- RAG implementation for financial document analysis
- Time series prediction with LSTM
- LLM integration for investment reasoning

## ✨ Key Features

### 🔥 Core Capabilities

- **Multi-Agent AI Committee**
  - 🔬 **Researcher Agent**: Gathers quantitative signals, news, and SEC filings
  - 🐂 **Bull Agent**: Generates bullish investment thesis
  - 🐻 **Bear Agent**: Presents bearish counterarguments
  - 💼 **Portfolio Manager**: Synthesizes all data into final recommendation (BUY/SELL/HOLD)

- **Advanced Analytics**
  - LSTM-based price movement prediction
  - Technical indicators (60-day moving average, volatility, returns)
  - Sentiment scoring from 10+ recent news articles
  - Vector-based semantic search through SEC 10-K filings

- **Interactive Dashboard**
  - Real-time Streamlit web interface
  - Visual metric cards with gradient designs
  - Bull vs Bear thesis comparison
  - Comprehensive decision tables

- **Automated Data Pipeline**
  - Automatic SEC filing download and indexing
  - FAISS vector storage for fast retrieval
  - Intelligent caching to avoid redundant downloads

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (app.py)                 │
│                  User Input → Results Visualization              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    LangGraph Orchestrator                        │
│                     (src/graph.py)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Researcher  │→ │  Bull Agent  │→ │   Portfolio  │          │
│  │    Node      │  └──────────────┘  │   Manager    │          │
│  │              │  ┌──────────────┐  │              │          │
│  └──────────────┘→ │  Bear Agent  │→ └──────────────┘          │
│                    └──────────────┘                              │
└────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│  Quant Tools   │  │ Sentiment Tools │  │   RAG System   │
│  (LSTM Model)  │  │ (Tavily Search) │  │  (SEC Filings) │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   yFinance     │  │  News Articles  │  │  FAISS Vector  │
│  Market Data   │  │   + Keywords    │  │     Store      │
└────────────────┘  └─────────────────┘  └────────────────┘
```

### 🔄 Data Flow

1. **User Input** → Ticker symbol (e.g., TSLA, NSE:RELIANCE)
2. **Researcher Node** → Fetches data from all sources in parallel
3. **Analyst Nodes** → Generate competing investment theses
4. **Portfolio Manager** → Makes final BUY/SELL/HOLD decision
5. **Dashboard** → Displays comprehensive analysis

## 🛠 Technology Stack

### AI/ML Framework
- **LangGraph** - Multi-agent orchestration framework
- **LangChain** - Tool integration and RAG pipeline
- **TensorFlow/Keras** - LSTM model training and inference
- **Groq (Llama 3.3 70B)** - Fast LLM inference for agent reasoning

### Data & Retrieval
- **FAISS** - Vector similarity search for RAG
- **HuggingFace Transformers** - Sentence embeddings (all-MiniLM-L6-v2)
- **yFinance** - Real-time market data
- **Tavily Search** - News and web search API
- **SEC Edgar API** - Automatic 10-K filing downloads

### Web Interface
- **Streamlit** - Interactive dashboard
- **Custom CSS** - Gradient cards and modern UI

### Data Processing
- **Pandas & NumPy** - Data manipulation
- **BeautifulSoup** - HTML parsing for SEC filings
- **RecursiveCharacterTextSplitter** - Document chunking

## 📥 Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager
- **Git**
- API keys for:
  - Groq AI
  - Tavily Search

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sreekarreddy2005/Hybrid-AI-Investment-System.git
   cd Hybrid-AI-Investment-System
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

   To get API keys:
   - **Groq**: Sign up at [console.groq.com](https://console.groq.com)
   - **Tavily**: Register at [tavily.com](https://tavily.com)

5. **Verify LSTM Model**
   
   Ensure `models/lstm_model.h5` exists in the models directory.

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```

7. **Access Dashboard**
   
   Open your browser to `http://localhost:8501`

## ⚙ Configuration

### Environment Variables (.env)

```env
# Required API Keys
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxx

# Optional Settings
MODEL_PATH=models/lstm_model.h5
VECTORSTORE_DIR=data/vectorstore
```

### Model Configuration

- **LSTM Model**: `models/lstm_model.h5`
  - Input: 60-day normalized price sequences
  - Output: Single prediction (bullish/bearish signal)
  - Architecture: Multi-layer LSTM with dropout

- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
  - Dimensions: 384
  - Used for SEC filing vector search

## 🚀 Usage

### Web Interface (Recommended)

1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Enter a ticker symbol in the sidebar:
   - **US Stocks**: TSLA, AAPL, MSFT, NVDA
   - **Indian Stocks**: NSE:RELIANCE, NSE:TCS

3. Click **⚡ ANALYZE**

4. View results:
   - Bull vs Bear analyst memos
   - Live metrics (price, returns, volatility)
   - LSTM prediction signal
   - News sentiment
   - Final BUY/SELL/HOLD recommendatio

## 📁 Project Structure

```
Hybrid-AI-Investment-System/
│
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── .env                            # API keys (create this)
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
├── README.md                       # This file
│
├── models/
│   └── lstm_model.h5              # Pre-trained LSTM for price prediction
│
├── src/
│   ├── agents/
│   │   ├── state.py               # AgentState TypedDict schema
│   │   ├── nodes.py               # Agent node implementations
│   │   └── graph.py               # LangGraph workflow definition
│   │
│   ├── tools/
│   │   ├── quant_tools.py         # LSTM prediction + technical indicators
│   │   ├── sentiment_tools.py     # News sentiment analysis
│   │   └── finance_tools.py       # yFinance fundamental data
│   │
│   ├── rag/
│   │   └── retriever.py           # SEC filing RAG retriever
│   │
│   └── ingestion/
│       ├── sec_downloader.py      # SEC Edgar API client
│       ├── processor.py           # HTML parsing + chunking
│       └── indexer.py             # FAISS vector store management
│
└── data/
    ├── filings/                   # Downloaded SEC 10-K HTML files
    └── vectorstore/               # FAISS indexes (auto-created)
        └── TSLA/                  # Per-ticker vector stores
```

## 🔧 How It Works

### Multi-Agent Collaboration

The system uses **LangGraph** to orchestrate four specialized AI agents:

1. **Researcher Agent**
   - Fetches real-time market data via yFinance
   - Downloads and indexes SEC 10-K filings
   - Gathers news sentiment from Tavily Search
   - Runs LSTM model for technical predictions

2. **Bull Agent**
   - Analyzes data with optimistic perspective
   - Identifies growth drivers and opportunities
   - Uses LLM (Llama 3.3 70B) to generate bullish thesis

3. **Bear Agent**
   - Evaluates same data with skeptical view
   - Highlights risks and challenges
   - Generates bearish counterarguments

4. **Portfolio Manager**
   - Synthesizes all agent outputs
   - Applies decision logic based on signals
   - Produces final BUY/SELL/HOLD recommendation

### Decision Logic

The Portfolio Manager uses a simple but effective rule:

- **BUY**: LSTM signal is Bullish AND News sentiment is Positive
- **SELL**: LSTM signal is Bearish AND News sentiment is Negative
- **HOLD**: Mixed or neutral signals

### RAG Pipeline (SEC Filings)

The system automatically:
1. Downloads latest 10-K filing from SEC Edgar
2. Parses HTML and chunks into 1000-character segments
3. Generates embeddings using HuggingFace transformers
4. Stores in FAISS vector database
5. Retrieves top-3 relevant chunks for analysis

**Caching**: Vector stores are saved locally to avoid re-downloading filings.

### Sentiment Analysis

Uses keyword-based scoring with two lists:

**Positive Keywords**: growth, beats, strong, profit, record, expansion, upgrade, bullish

**Negative Keywords**: decline, miss, weak, loss, downgrade, lawsuit, risk, bearish, crash

The system counts keyword occurrences across 10 recent news articles and calculates a net sentiment score.

## 📈 Example Output

### Sample Analysis for TSLA

```
🎯 BUY Recommendation

📊 Live Metrics
─────────────────────────────────────
💰 Price         | $242.84
📈 30D Return    | +12.3%
⚡ Volatility   | 3.45
🎯 LSTM Signal   | Bullish
📰 Sentiment     | Positive (8 articles)

🐂 Bull Case
• Strong delivery growth in Q4 2024 exceeding expectations
• Energy storage revenue up 120% YoY
• FSD Beta showing significant improvements
• Cybertruck production ramping ahead of schedule

🐻 Bear Case
• Increased competition in EV market from legacy automakers
• Regulatory scrutiny on Autopilot safety
• High valuation metrics (P/E ratio above industry)
• Dependence on CEO's public image

🤖 AI Analysis
LSTM: Bullish prediction based on recent uptrend
Sentiment: Positive from 8 news articles
SEC Filings: Company highlights risks including supply chain constraints,
             but emphasizes long-term growth in sustainable energy
Consensus: Bull case favors growth momentum vs bearish valuation concerns
```

## 🔑 API Keys Required

### 1. Groq API (Required)

**What it's for**: LLM inference for Bull/Bear/PM agents

**How to get**:
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up (free tier available)
3. Create API key
4. Add to `.env`: `GROQ_API_KEY=gsk_...`

**Model Used**: `llama-3.3-70b-versatile`

### 2. Tavily API (Required)

**What it's for**: News search and sentiment data

**How to get**:
1. Visit [tavily.com](https://tavily.com)
2. Sign up (free tier: 1000 searches/month)
3. Get API key
4. Add to `.env`: `TAVILY_API_KEY=tvly-...`

### 3. SEC Edgar (No Key Required)

**What it's for**: Automatic 10-K filing downloads

**Rate Limits**: 10 requests/second (handled automatically)

## 🐛 Troubleshooting

### Common Issues

**1. "LSTM model not loaded"**

```bash
# Verify model file exists
ls models/lstm_model.h5
```

**2. "GROQ_API_KEY not found"**

```bash
# Check .env file exists
cat .env

# Verify key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('GROQ_API_KEY'))"
```

**3. "SEC request failed (403)"**

- Update the email in `src/ingestion/sec_downloader.py`
- Replace `student@example.com` with your actual email

**4. "Insufficient historical data"**

- Stock needs 60+ trading days of history
- Try a more established ticker (e.g., AAPL instead of recent IPO)

**5. Vector store errors**

```bash
# Clear cached indexes
rm -rf data/vectorstore/

# Re-run analysis to rebuild
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

### Potential Enhancements

- [ ] Add XGBoost/Random Forest ensemble
- [ ] Implement portfolio optimization (multi-ticker)
- [ ] Add backtesting framework
- [ ] Support more data sources (Polygon.io, IEX Cloud)
- [ ] Fine-tune LSTM on more recent data
- [ ] Add risk metrics (Sharpe ratio, max drawdown)
- [ ] Support cryptocurrency analysis
- [ ] Add export to PDF reports

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Commit (`git commit -m 'Add amazing feature'`)
5. Push (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify
- ✅ Distribute
- ✅ Private use

Conditions:
- Include original license
- State changes made

## ⚠ Disclaimer

### IMPORTANT - READ CAREFULLY

This software is provided for **EDUCATIONAL AND RESEARCH PURPOSES ONLY**.

#### ⚠️ Investment Risks

- **Past performance ≠ future results**
- AI predictions can be **wrong**
- Markets are **unpredictable**
- You can **lose money**

#### 🚫 No Warranties

The creators provide this software **"AS IS"** with:
- ❌ No accuracy guarantees
- ❌ No profit guarantees  
- ❌ No liability for losses
- ❌ No warranty of any kind

#### 📚 Educational Value

This project demonstrates:
- Multi-agent AI systems
- RAG for financial documents
- Time series prediction
- LLM integration

**Use it to learn, not to trade.**

---

## 📧 Contact & Support

**Developer**: Sreekar Reddy

- 🐙 GitHub: [@Sreekarreddy2005](https://github.com/Sreekarreddy2005)
- 📂 Repository: [Hybrid-AI-Investment-System](https://github.com/Sreekarreddy2005/Hybrid-AI-Investment-System)
- 💬 Issues: [GitHub Issues](https://github.com/Sreekarreddy2005/Hybrid-AI-Investment-System/issues)

## 🙏 Acknowledgments

### Technologies
- **LangChain** & **LangGraph** - Agent orchestration
- **Groq** - Ultra-fast LLM inference
- **Tavily** - Real-time search API
- **Streamlit** - Beautiful dashboards
- **FAISS** - Efficient vector search

### Data Sources
- **SEC Edgar** - Public company filings
- **yFinance** - Market data
- **HuggingFace** - Embedding models

### Community
- Quantitative finance researchers
- Open-source AI/ML community
- LangChain developers

---

<div align="center">

### ⭐ If you find this project helpful, please star the repository!

**Built with ❤️ by Sreekar Reddy Pindi**

*Demonstrating the future of AI-powered investment research*

</div>
