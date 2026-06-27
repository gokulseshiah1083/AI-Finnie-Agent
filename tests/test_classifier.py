import pytest
from src.workflow.classifier import classify_query


@pytest.mark.parametrize("query,expected", [
    ("What is a stock?", "finance_qa"),
    ("How does compound interest work?", "finance_qa"),
    ("Analyze my portfolio please", "portfolio_analysis"),
    ("My holdings are 50% stocks and 50% bonds", "portfolio_analysis"),
    ("What is the price of AAPL?", "market_analysis"),
    ("How is SPY doing today?", "market_analysis"),
    ("I want to save for retirement in 20 years", "goal_planning"),
    ("How much do I need to save to buy a house?", "goal_planning"),
    ("What's in the financial news today?", "news_synthesizer"),
    ("What are the latest market headlines?", "news_synthesizer"),
    ("What is a Roth IRA?", "tax_education"),
    ("How does tax-loss harvesting work?", "tax_education"),
    ("Explain 401k contribution limits", "tax_education"),
])
def test_classifier_routing(query, expected):
    result = classify_query(query)
    assert result == expected, f"Query: '{query}' → expected {expected}, got {result}"
