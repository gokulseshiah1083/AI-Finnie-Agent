# Architecture Diagram

Three zoom levels (C4-style: context → container → the turn loop), plus the data + market-data
fallback path. Diagrams are Mermaid so they version-control as text and render in-place. This is the
visual companion to [DESIGN.md](DESIGN.md), which carries the rationale for every choice shown here.

## L1 — System context

```mermaid
flowchart LR
    user([Beginner investor]):::actor

    subgraph FN[AI Finnie Agent]
        core[Multi-agent finance educator]
    end

    subgraph KNOW[Internal knowledge]
        kb[(Finance knowledge base<br/>FAISS — embedded)]
    end

    llm[Claude API<br/>single model, config-driven]:::ext
    av[Alpha Vantage<br/>quotes + news]:::ext
    yf[yfinance<br/>fallback]:::ext

    user -->|plain-English question| core
    core -->|educational answer + disclaimer| user
    core -->|semantic retrieval| kb
    core -->|explanation + interpretation| llm
    core -->|live quotes / news| av
    core -.->|on AV miss / no key| yf

    classDef actor fill:#dbeafe,stroke:#1e40af,color:#111;
    classDef ext fill:#f1f5f9,stroke:#475569,color:#111;
```

## L2 — Container view

```mermaid
flowchart TB
    user([Beginner investor]):::actor

    subgraph UI[Interface tier]
        st[Streamlit app<br/>lazy-loaded engines, config-driven branding]
        tabs[Tabs: Chat / Portfolio / Market / Goals]
    end

    subgraph WF[LangGraph workflow]
        clf[classifier<br/>deterministic keyword scoring]
        route{route by score<br/>default finance_qa}
        append[append_message]
    end

    subgraph AG[6 specialist agents — one LLM call each]
        rag_ag[RAG agents<br/>finance_qa, tax_education]
        calc_ag[Compute agents<br/>portfolio, goal_planning]
        live_ag[Live-data agents<br/>market, news]
    end

    subgraph SRC[Per-agent evidence sources]
        kb[(FAISS KB<br/>MiniLM, whole-doc, top_k=3)]
        pymath[Python metrics<br/>diversification, goal projection]
        md[Market-data layer<br/>Alpha Vantage → yfinance, 60m cache]
    end

    subgraph PLAT[Platform]
        llmc[LLM client<br/>single model, alias-resilient, graceful fallback]
        trace[Tracing<br/>turn → node → llm spans, JSONL]
    end

    llm[Claude API]:::ext

    user --> st --> tabs --> clf
    clf --> route
    route --> rag_ag & calc_ag & live_ag
    rag_ag --> kb
    calc_ag --> pymath
    live_ag --> md
    rag_ag & calc_ag & live_ag --> llmc --> llm
    rag_ag & calc_ag & live_ag --> append
    clf & rag_ag & calc_ag & live_ag --> trace

    classDef actor fill:#dbeafe,stroke:#1e40af,color:#111;
    classDef ext fill:#f1f5f9,stroke:#475569,color:#111;
```

## L3 — The turn loop (a single question)

```mermaid
sequenceDiagram
    autonumber
    participant U as Investor
    participant C as classifier
    participant A as Agent (1 of 6)
    participant E as Evidence source
    participant L as LLM client
    participant M as append_message

    U->>C: user_query + conversation history
    C->>C: score regex rules, pick max (default finance_qa)
    C->>A: route to agent_type
    alt RAG agent (finance_qa, tax)
        A->>E: FAISS retrieve, optional category filter
        E-->>A: cited context [Source: title]
    else Compute agent (portfolio, goals)
        A->>E: Python computes metrics / projection
        E-->>A: exact numbers as JSON
    else Live-data agent (market, news)
        A->>E: Alpha Vantage, fall back to yfinance
        E-->>A: quotes / news, source labeled
    end
    A->>L: system prompt + injected evidence + last ~10 turns
    L-->>A: explanation (graceful fallback string on API error)
    A->>A: append educational disclaimer
    A->>M: response
    M-->>U: answer + disclaimer
    Note over A,L: agents interpret evidence, never recompute the numbers
```

## Knowledge base & market-data fallback path

```mermaid
flowchart LR
    docs[Knowledge-base JSON<br/>title + content + category]:::src

    subgraph BUILD[KB build / load]
        emb[Embed whole docs<br/>MiniLM, normalized]
        faiss[(FAISS IndexFlatIP<br/>persisted index + docs.pkl)]
    end

    subgraph LIVE[Market-data layer]
        cache{fresh in<br/>60m cache?}
        av[Alpha Vantage<br/>GLOBAL_QUOTE, OVERVIEW, NEWS]
        yfb[yfinance fallback]
        gk[General-knowledge fallback<br/>if both unavailable]
    end

    docs --> emb --> faiss
    cache -->|yes| hit[return cached]
    cache -->|no| av
    av -->|miss / no API key| yfb
    yfb -->|miss| gk

    classDef src fill:#f1f5f9,stroke:#475569,color:#111;
```

**Reading the diagrams.** Two ideas are visible structurally:

- **Three agent patterns behind one interface** (L2 `AG` subgraph; L3 `alt` branches). *RAG* agents cite
  the knowledge base, *compute* agents run exact Python math the LLM only narrates, and *live-data*
  agents pull real quotes/news with a fallback chain. See [DESIGN.md §2](DESIGN.md) and [§3–5](DESIGN.md).
- **Compute in code, narrate with the model** (L3 note; the `pymath` node). Portfolio metrics and goal
  projections are computed deterministically and the LLM is instructed to interpret, not recompute —
  the project's core defence against confidently-wrong numbers. See [DESIGN.md §3](DESIGN.md).

Two safety properties are also structural: **graceful degradation everywhere** — Alpha Vantage →
yfinance → general knowledge, and an LLM-error fallback string, so a missing API key or provider
outage never hard-fails a turn ([DESIGN.md §5](DESIGN.md)); and an **educational-only posture** — an
advice disclaimer is appended to every answer and agents are prompted away from buy/sell calls
([DESIGN.md §7](DESIGN.md)).
