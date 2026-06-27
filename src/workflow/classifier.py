import re

ROUTING_RULES = [
    ("portfolio_analysis", [
        r"\bportfolio\b", r"\bholdings?\b", r"\bstocks?\s+i\s+(have|own|hold)\b",
        r"\d+%\s*(stocks?|bonds?|etfs?)\b",
        r"\bdiversif", r"\ballocation\b", r"\brebalance\b", r"\bmy\s+investment\b",
        r"\bexpense\s+ratio\b", r"\banalyze\s+my\b",
    ]),
    ("market_analysis", [
        r"\bprice\b", r"\bstock\s+price\b", r"\bmarket\s+(today|now|current|trend)\b",
        r"\btrading\b", r"\bquote\b", r"\bperformance\b", r"\bhow\s+is\b.*\bdoing\b",
        r"\b[A-Z]{2,5}\b\s+(stock|share|price|etf)", r"\bbull\b|\bbear\b",
        r"\bindex\b", r"\bs&p\b|\bnasdaq\b|\bdow\b", r"\bspy\b|\bqqq\b|\bvti\b",
    ]),
    ("goal_planning", [
        r"\bgoal\b", r"\bplan\b", r"\bretire\b", r"\bsave\s+for\b",
        r"\bhow\s+(much|long|many)\b.*(save|invest|need)\b",
        r"\bdown\s+payment\b", r"\bcollege\b|\beducation\s+fund\b",
        r"\bfinancial\s+(freedom|independence)\b", r"\bmillion\b",
        r"\bproject(ion)?\b", r"\byears\s+(to|until)\b",
    ]),
    ("news_synthesizer", [
        r"\bnews\b", r"\bheadlines?\b", r"\btoday\b", r"\brecent\b",
        r"\bwhat('s|\s+is)\s+happening\b", r"\bmarket\s+update\b",
        r"\bearnings\b", r"\bfed\b|\bfederal\s+reserve\b", r"\brate\s+(hike|cut)\b",
    ]),
    ("tax_education", [
        r"\btax(es|able)?\b", r"\b401k\b|\b401\(k\)\b", r"\bira\b|\broth\b",
        r"\bcapital\s+gains?\b", r"\bdividend\s+tax\b", r"\btax[\s-]loss\b",
        r"\bwash\s+sale\b", r"\btax[\s-]advantaged\b", r"\bdeductible\b",
    ]),
]


def classify_query(query: str) -> str:
    q = query.lower()
    scores: dict[str, int] = {}

    for agent, patterns in ROUTING_RULES:
        score = sum(1 for p in patterns if re.search(p, q))
        if score > 0:
            scores[agent] = score

    if not scores:
        return "finance_qa"

    return max(scores, key=scores.get)
