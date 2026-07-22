# Technical Design

Design decisions, why they were made, and what they cost. Companion to the README, which covers
usage, and to [ARCHITECTURE.md](ARCHITECTURE.md), which carries the diagrams.

---

## 1. System architecture

### 1.1 One classifier, six specialists, one call per turn

The graph is deliberately shallow: `classifier → one of six agents → append_message → END`. Each
agent answers in a **single LLM call** with its evidence pre-fetched and injected — there is no
multi-step tool loop. For a beginner-education assistant that explains rather than investigates, the
work is *fetch the right evidence, then explain it well*, and that maps to one grounded call. The six
agents mirror the product's capabilities: `finance_qa`, `portfolio_analysis`, `market_analysis`,
`goal_planning`, `news_synthesizer`, `tax_education`.

Cost: a genuinely multi-step question ("analyze my portfolio *and* pull today's price for each
holding") is not chained across agents automatically; it is handled within the routed agent's
evidence step or split by the user. Deliberate — the simplicity buys testability and predictable
latency.

### 1.2 Deterministic routing, not an LLM router

Routing is a **keyword-scoring classifier** (`classifier.py`): each agent owns a list of regex
patterns, the query scores against all of them, and the highest score wins, defaulting to
`finance_qa` when nothing matches. Consequences:

- Routing is unit-testable with **no API key** — it is a pure function of the query string.
- A routing miss degrades to the general Q&A agent, never to a crash.
- Route decisions are explainable: they are a function of which keywords matched.

Cost: keyword routing can misroute genuinely ambiguous phrasing (e.g. a "tax" question that never
says "tax"). An LLM router would catch more nuance at the price of a model call, non-determinism, and
untestability on every turn — a poor trade for a front-door classifier that must be cheap and stable.

---

## 2. Agent design

Three patterns sit behind one `BaseAgent` interface. Each agent overrides how it gathers evidence;
all of them end at the same `call_llm` and inherit the same disclaimer and error handling.

| Agent | Pattern | Evidence | LLM role |
|---|---|---|---|
| `finance_qa` | RAG | FAISS retrieval over the knowledge base | Explain + cite |
| `tax_education` | RAG | FAISS retrieval, `category="tax"` filter | Explain + cite |
| `portfolio_analysis` | Compute | Python metrics from user holdings | Interpret metrics |
| `goal_planning` | Compute | Python future-value projection | Interpret projection |
| `market_analysis` | Live data | Alpha Vantage → yfinance quotes | Explain the numbers |
| `news_synthesizer` | Live data | Alpha Vantage → yfinance news | Summarize + contextualize |

`BaseAgent.run` wraps `_process` in a try/except: on any failure it returns the configured
`fallback_message` and records the error in state, so one agent's bad day is a graceful message, not a
stack trace to the user. History is trimmed to the last ~10 turns before every call to bound context
cost.

---

## 3. Compute in code, narrate with the model

The single most consequential decision, and the project's main defence against confident wrong
answers: **numbers are computed in Python; the LLM only interprets them.**

- `portfolio_analysis` computes total value, per-holding allocation, asset-class breakdown, and a
  diversification score in `_analyze_portfolio` — then hands the LLM the finished metrics as JSON.
- `goal_planning` computes the future-value projection, shortfall, and required monthly contribution
  in `project_goal` — closed-form compound-interest math — then hands the LLM the result.

The system prompts instruct the agents to *interpret* the provided figures, not to recompute them. An
LLM asked to do compound-interest arithmetic in prose will sometimes be confidently wrong; a Python
function will not. This mirrors the companion billing project's "report the numbers the tools
returned, never recompute them" rule.

Cost: the compute agents can only answer what their functions express. A novel analytical question
needs new code, not just a new prompt. Acceptable for a system whose numbers must be trustworthy.

---

## 4. RAG implementation

### 4.1 Whole-document embeddings, cited

The knowledge base is embedded with `all-MiniLM-L6-v2` into a FAISS `IndexFlatIP` (inner product over
normalized vectors ≈ cosine). Documents are embedded **whole** (`title\ncontent`), not chunked — the
entries are short, self-contained finance explainers, and splitting one mid-concept would retrieve
half an idea. Retrieval returns the top `k` (default 3) as `[Source: <title>]` blocks, and the
`finance_qa` prompt instructs the model to cite them.

### 4.2 Category filter with fallback

`tax_education` retrieves with `category="tax"` so tax questions pull tax documents. If the filter
yields nothing, retrieval **falls back to the unfiltered top-k** rather than returning empty — a
missing category should degrade to "less targeted," never to "no context." If no index exists at all,
`retrieve` returns an empty string and the agent still answers from the model's general knowledge.

Cost: whole-doc embedding assumes a curated, short-document corpus; a corpus of long documents would
need chunking to keep retrieval precise.

---

## 5. Live market data, with graceful degradation

Market and news evidence come from a layered source in `market_data.py`:

1. **Alpha Vantage** (primary) — `GLOBAL_QUOTE`, `OVERVIEW`, `TIME_SERIES_DAILY`, `NEWS_SENTIMENT`.
2. **yfinance** (fallback) — used automatically when Alpha Vantage misses, errors, or no API key is
   set. Every result is **source-labeled** (`"Alpha Vantage"` vs `"yfinance (fallback)"`) so the
   provenance of a number is never ambiguous.
3. **General-knowledge fallback** — if both live sources are unavailable, the agent is told to answer
   from general knowledge rather than fail.

A 60-minute TTL cache fronts both providers so repeated questions about the same ticker don't re-hit
the network, and the free-tier rate limits last longer. Ticker extraction from free text is a regex
with a common-word stop-list (so "MY", "ETF", "AI" aren't treated as symbols).

Cost: the common-word heuristic can still miss or over-match unusual tickers; free-tier Alpha Vantage
omits some fields (e.g. yearly change), which the code returns as `None` rather than faking.

---

## 6. Model access

`call_llm` uses a **single configured model** (`llm.model`, currently `claude-opus-4-8`) with a
`max_tokens` from config, wrapped in a trace span that records token usage and stop reason. Two
deliberate touches:

- **Alias resilience.** `resolve_model_name` maps legacy model ids (older Haiku/Sonnet names) forward
  to the current model, so a stale config value can't break a request.
- **Graceful failure.** Any exception from the API returns a friendly fallback string rather than
  propagating — consistent with the "never hard-fail a turn" posture.

This is intentionally simpler than the companion billing project's effort-tiered, family-branching
client: Finnie's agents differ in *evidence*, not in reasoning depth, so one model with good prompts
fits. If cost or latency later demands tiering, the config-driven client is where it would go.

---

## 7. Educational-only posture (the safety boundary)

Finnie gives **financial education, not financial advice**, and that boundary is enforced
structurally rather than hoped for:

- `BaseAgent.run` appends an educational disclaimer to **every** response — there is no path that
  returns an answer without it.
- Agent prompts explicitly avoid buy/sell recommendations: the market agent "discusses trends without
  making specific buy/sell recommendations," the tax agent defers to a CPA for personal situations.

Cost: the assistant will decline to tell a user what to buy — which is the point. The disclaimer is
redundant on obviously-educational answers, but redundancy on a compliance boundary is a feature.

---

## 8. State and tracing

### 8.1 State

`AgentState` is a `TypedDict` carrying `messages` (via LangGraph's `add_messages`), `user_query`,
`agent_type`, `rag_context`, `portfolio_data`, `market_data`, `response`, `session_id`, and `error`.
As in any LangGraph app, **every field the graph carries must be declared** — an undeclared key is
silently dropped and reads back as its default inside every node.

### 8.2 Tracing

Every turn writes a tree of spans — `turn → node:<name> → llm` — as one JSON line per span to
`logs/trace.jsonl`, with token counts, stop reason, and the chosen route. Instrumentation is added
once, at node registration (`_traced`), so nodes stay testable without the tracing layer. Inspect with
the Developer-traces expander in the UI or `jq -s 'group_by(.trace_id)' logs/trace.jsonl`.

---

## 9. Interface & operations

- **Streamlit UI**, config-driven branding (every label reads from `config.yaml`), with tabs for
  Chat, Portfolio, Market, and Goals.
- **Lazy loading.** The page shell paints instantly; the heavy engines (LangGraph, agents,
  sentence-transformers, FAISS, market data) load on first AI action or via a preload button — so a
  cold visitor isn't blocked on a model download to see the page.
- **Config over code.** Model, RAG parameters, market-data providers, cache TTL, tracing, and all UI
  copy live in `config.yaml`, so behaviour and voice change without touching code.

---

## 10. Known limitations

- **Single-process session state.** Conversation and portfolio state live in the Streamlit session;
  horizontal scaling needs an external store.
- **Keyword routing.** Ambiguous phrasing can misroute; the `finance_qa` default is the safety net.
- **Single LLM call per turn.** No automatic multi-step chaining across agents; a compound question
  is handled within one agent's evidence step or split by the user.
- **Live data is API-key- and rate-limit-gated.** Without an Alpha Vantage key the system uses
  yfinance, and under free-tier limits some fields are absent (returned as `null`, never faked).
- **Synthetic/curated knowledge base.** Modelled on real finance concepts but not authoritative;
  production would source from vetted educational material.
- **No authentication.** The UI has no auth layer; any real deployment needs one in front.
- **Not financial advice.** By design — the system educates and refuses personalized buy/sell or
  personal-tax guidance.
