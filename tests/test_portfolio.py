import pytest
from src.agents.portfolio_agent import _analyze_portfolio


SAMPLE_HOLDINGS = [
    {"ticker": "VTI", "shares": 50, "current_price": 240, "asset_type": "ETF"},
    {"ticker": "BND", "shares": 20, "current_price": 73, "asset_type": "Bond/Fixed Income"},
    {"ticker": "AAPL", "shares": 10, "current_price": 195, "asset_type": "US Stock"},
]


def test_portfolio_metrics_basic():
    metrics = _analyze_portfolio(SAMPLE_HOLDINGS)
    assert metrics["total_value"] == pytest.approx(15410.0, rel=1e-3)
    assert metrics["num_holdings"] == 3
    assert 0 <= metrics["diversification_score"] <= 100


def test_portfolio_allocation_sums_to_100():
    metrics = _analyze_portfolio(SAMPLE_HOLDINGS)
    total_pct = sum(h["percentage"] for h in metrics["holdings"])
    assert abs(total_pct - 100.0) < 0.1


def test_empty_portfolio():
    metrics = _analyze_portfolio([])
    assert metrics == {}


def test_single_holding_low_diversification():
    holdings = [{"ticker": "AAPL", "shares": 100, "current_price": 200, "asset_type": "US Stock"}]
    metrics = _analyze_portfolio(holdings)
    assert metrics["diversification_score"] < 70
