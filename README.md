# AI Finance Assistant

A production-ready multi-agent financial education system powered by Claude Sonnet and LangGraph.

## Architecture

```
User Query → LangGraph Classifier → Specialized Agent → RAG + LLM → Response
```

### 6 Specialized Agents

| Agent | Purpose |
|-------|---------|
| Finance Q&A | General financial education with RAG |
| Portfolio Analysis | Holdings metrics, diversification scoring |
| Market Analysis | Real-time stock data via yfinance |
| Goal Planning | Financial goal projections with compound interest |
| News Synthesizer | Financial news summarization |
| Tax Education | Tax concepts, 401k/IRA, tax-loss harvesting |

### Tech Stack

- **LLM**: Claude Sonnet (`claude-sonnet-4-6`) via Anthropic SDK
- **Orchestration**: LangGraph `StateGraph`
- **RAG**: FAISS + `sentence-transformers` (all-MiniLM-L6-v2)
- **Market Data**: `yfinance` (free, no API key required)
- **UI**: Streamlit (4 tabs: Chat, Portfolio, Market, Goals)
- **Knowledge Base**: 20 curated financial education articles

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key
```bash
cp .env.example .env
# Edit .env and add your key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the app
```bash
python run.py
# or directly:
streamlit run src/web_app/app.py
```

### 4. Run tests
```bash
pytest tests/ -v
```

### 5. Deploy to Streamlit Community Cloud
- Push this repository to GitHub.
- In Streamlit Cloud, create a new app from the repo.
- Set the main file to `app.py` or `streamlit_app.py`.
- Add these secrets in the Streamlit Cloud app settings:
  - `ANTHROPIC_API_KEY`
  - `ALPHA_VANTAGE_API_KEY` (optional)

## Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/          # 6 specialized agents
│   ├── core/            # State, LLM client, base agent, config
│   ├── data/            # Knowledge base JSON articles
│   ├── rag/             # FAISS vector store + retrieval
│   ├── web_app/         # Streamlit multi-tab interface
│   ├── utils/           # Market data (yfinance), logging
│   └── workflow/        # LangGraph graph + query classifier
├── tests/               # Unit tests (23 tests, 100% pass)
├── config.yaml          # Centralized configuration
├── requirements.txt
└── README.md
```

## Disclaimer

This tool is for **educational purposes only** and does not constitute financial advice.
Always consult a qualified financial advisor before making investment decisions.