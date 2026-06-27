import pytest
from src.agents.goal_planning_agent import project_goal


def test_on_track_goal():
    result = project_goal(
        goal_amount=50000,
        current_savings=10000,
        monthly_contribution=500,
        annual_return_pct=7,
        years=10,
    )
    assert result["on_track"] is True
    assert result["projected_total"] > 50000
    assert result["shortfall"] == 0


def test_shortfall_goal():
    result = project_goal(
        goal_amount=1_000_000,
        current_savings=1000,
        monthly_contribution=100,
        annual_return_pct=7,
        years=5,
    )
    assert result["on_track"] is False
    assert result["shortfall"] > 0
    assert result["required_monthly_contribution"] > 100


def test_zero_return():
    result = project_goal(
        goal_amount=12000,
        current_savings=0,
        monthly_contribution=1000,
        annual_return_pct=0,
        years=1,
    )
    assert result["projected_total"] == pytest.approx(12000.0, rel=1e-2)
    assert result["on_track"] is True


def test_years_to_goal_calculated():
    result = project_goal(
        goal_amount=100000,
        current_savings=0,
        monthly_contribution=500,
        annual_return_pct=7,
        years=30,
    )
    assert result["years_to_goal_at_current_rate"] is not None
    assert result["years_to_goal_at_current_rate"] > 0
