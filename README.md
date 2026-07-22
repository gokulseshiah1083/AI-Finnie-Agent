# AI Finnie Agent

AI Finnie Agent is a multi-agent financial education platform designed to help users explore investing, portfolio analysis, market trends, financial planning, and tax concepts through a conversational interface. Built with Python, Streamlit, LangGraph, and large language models, it combines structured financial reasoning with retrieval-augmented guidance.

Finnie shares its design system with the companion
[AI-Billing-Assistance](https://github.com/gokulseshiah1083/AI-Billing-Assistance)
project: the same service-desk styling, config-driven branding, lazy loading,
and execution tracing — so both codebases read, look, and debug the same way.

## Overview

The system is organized around a simple principle: route each user question to the most appropriate specialist, then enrich the response with the right context. This allows the assistant to provide targeted answers instead of relying on a single general-purpose model for every request.

## Design system (shared with AI-Billing-Assistance)

**Service-desk UI.** The app presents as a finance desk: a light theme with a
soft off-white page and bank-blue accents (`.streamlit/config.toml`), gradient
section banners on every view (`src/web_app/components.py`), and a sidebar
identity block — "🕵️ AI Finnie Agent, Personal Finance Analyst" — above a
config-driven list of the six agents. Every label, tab name and tagline lives
under `streamlit.branding` in `config.yaml`, so the voice is re-wordable
without touching code.

**Navigation** is a segmented control rather than `st.tabs`, driven by session
state — visually the same, but programmatically switchable (`pending_view`),
which the billing project uses for flow redirects.

**Chat, newest-first.** Popular questions are pinned at the top of Ask Finnie
and stay clickable after every response; the input box sits directly beneath
them and never moves; history renders latest-turn-first, so you never scroll
to ask the next question.

**Lazy loading with an explicit preload.** The page shell paints instantly;
the heavy stack (LangGraph, sentence-transformers/torch, FAISS, market data)
loads on the first AI action or via the "⚡ Load AI engines now" button at the
top of the page. Configured under `streamlit.lazy_load`.

**Execution tracing.** Every query turn writes a span tree to
`logs/trace.jsonl` — `turn` → `node:classifier` / `node:<agent>` → `llm`
(with token counts) — one JSON line per span, each carrying `trace_id`,
`parent_span_id`, `duration_ms` and `status`. Dependency-free
(`src/core/trace.py`), fail-safe (a broken trace path never breaks an
answer), configured under `tracing:`, and viewable in the **Developer traces**
expander on the Welcome Desk or with
`jq -s 'group_by(.trace_id)' logs/trace.jsonl`.

## Architecture

```text
User Query
  ↓
LangGraph Orchestrator
  ↓
Query Classifier
  ↓
Specialized Agent
  ↓
RAG / Live Market Data / Portfolio Metrics
  ↓
LLM Response
  ↓
Streamlit Interface
```

## Documentation

Deeper design docs live alongside this README:

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — C4-style Mermaid diagrams: system context,
  container view, the per-turn agent loop, and the knowledge-base / market-data
  fallback path.
- **[DESIGN.md](DESIGN.md)** — the technical design in "decision → why → what it
  cost" form: routing, the three agent patterns (RAG, deterministic compute, live
  data), grounding, graceful degradation, and the educational-only safety boundary.

## Agent Overview

| Agent | What it does | Sample prompt |
|-------|--------------|---------------|
| Finance Q&A | Explains general financial concepts using the knowledge base and retrieval context. | "What is the difference between a Roth IRA and a traditional IRA?" |
| Portfolio Analysis | Evaluates holdings, asset allocation, diversification, and portfolio balance. | "Analyze my portfolio and tell me whether I am diversified enough." |
| Market Analysis | Provides market context, stock-related insights, and live market data interpretation. | "How is Apple performing today, and what does its P/E ratio mean?" |
| Goal Planning | Helps users estimate how much they need to save to reach future financial goals. | "I want to save $100,000 for a house in 7 years. How much should I save each month?" |
| News Synthesizer | Summarizes recent financial news and explains why it matters to investors. | "Summarize today’s market headlines and explain their impact in simple terms." |
| Tax Education | Explains tax concepts such as capital gains, dividends, and tax-advantaged accounts. | "How does tax-loss harvesting work?" |

## Workflow

The workflow is designed to be clear and modular:

1. A user submits a question through the Streamlit interface.
2. The LangGraph orchestrator passes the query to the classifier.
3. The classifier routes the request to the most relevant agent based on keywords and intent.
4. The selected agent gathers context from the knowledge base, portfolio data, live market data, or other structured inputs.
5. The LLM is invoked with that context and the user prompt.
6. The response is returned to the user with the appropriate agent context.

### Example: Finance question flow

When a user asks, "What is compound interest?", the orchestrator routes the request to the Finance Q&A agent. That agent retrieves relevant financial education content, sends the prompt and context to the LLM, and returns a clear explanation to the user.

## Tech Stack

- **Language**: Python
- **UI**: Streamlit
- **Workflow Orchestration**: LangGraph
- **LLM Integration**: Anthropic model via the configured LLM client
- **RAG**: FAISS + sentence-transformers
- **Market Data**: yfinance and other live data sources
- **Visualization**: Plotly and pandas
- **Testing**: pytest

## Project Structure

```text
AI-Finance-Assistance/
├── app.py                # Entry point for the Streamlit experience
├── run.py                # Run helper
├── config.yaml           # Central configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── agents/           # Specialized agent implementations
│   ├── core/             # LLM, state, and shared agent logic
│   ├── data/             # Knowledge base assets
│   ├── rag/              # Retrieval and knowledge base integration
│   ├── utils/            # Market data and utility helpers
│   ├── web_app/          # Streamlit UI components
│   └── workflow/         # Query classifier and LangGraph graph
├── tests/                # Unit and integration tests
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Set your API key for the LLM provider and any optional market data credentials:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

If you are deploying through Streamlit Cloud, add the same values as secrets in your app settings.

### 3. Run the application

```bash
python run.py
```

Or launch the UI directly:

```bash
streamlit run src/web_app/app.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Deployment

To deploy the app to Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new app in Streamlit Cloud.
3. Set the main file to `app.py` or `streamlit_app.py`.
4. Add the required secrets:
   - `ANTHROPIC_API_KEY`
   - `ALPHA_VANTAGE_API_KEY` (optional)

## Disclaimer

This application is intended for educational purposes only. It does not constitute financial advice. Always consult a qualified financial professional before making investment or tax decisions.