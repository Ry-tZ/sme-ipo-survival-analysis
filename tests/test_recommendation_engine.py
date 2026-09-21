"""
Unit tests for the SME IPO Recommendation & Allotment Engine.
"""

import pytest
import numpy as np
from src.models.recommendation_engine import (
    get_sebi_lot_size,
    get_minimum_investment,
    calculate_allotment_probabilities,
    calculate_capital_allocation,
    SMERecommendationEngine
)


def test_sebi_lot_size_slabs():
    """Verify SEBI statutory lot sizes across various price points."""
    assert get_sebi_lot_size(10.0) == 10000
    assert get_sebi_lot_size(14.0) == 10000
    assert get_sebi_lot_size(16.0) == 8000
    assert get_sebi_lot_size(24.0) == 6000
    assert get_sebi_lot_size(30.0) == 4000
    assert get_sebi_lot_size(45.0) == 3000
    assert get_sebi_lot_size(65.0) == 2000
    assert get_sebi_lot_size(85.0) == 1600
    assert get_sebi_lot_size(95.0) == 1200
    assert get_sebi_lot_size(143.0) == 1000
    assert get_sebi_lot_size(175.0) == 800
    assert get_sebi_lot_size(220.0) == 600
    assert get_sebi_lot_size(300.0) == 400
    assert get_sebi_lot_size(450.0) == 300
    assert get_sebi_lot_size(550.0) == 240
    assert get_sebi_lot_size(680.0) == 200
    assert get_sebi_lot_size(850.0) == 160
    assert get_sebi_lot_size(1200.0) == 100


def test_minimum_investment_calculation():
    """Ensure lot sizes keep minimum ticket size within ~INR 1.0L to 1.5L."""
    res1 = get_minimum_investment(95.0)
    assert res1['lot_size'] == 1200
    assert res1['min_amount_inr'] == 114000.0

    res2 = get_minimum_investment(143.0)
    assert res2['lot_size'] == 1000
    assert res2['min_amount_inr'] == 143000.0

    # Custom lot size override
    res3 = get_minimum_investment(100.0, lot_size=2000)
    assert res3['lot_size'] == 2000
    assert res3['min_amount_inr'] == 200000.0


def test_allotment_probabilities_retail():
    """Verify statutory computerized lottery probabilities."""
    # Highly oversubscribed (100x)
    probs_100x = calculate_allotment_probabilities(100.0, max_pans=5)
    assert probs_100x['p_retail_single_pct'] == 1.0
    assert probs_100x['p_retail_ratio'] == "1 in 100"
    
    # 5 PANs: 1 - (1 - 0.01)^5 = 1 - 0.95099 = 0.04901 (4.90%)
    pan5 = probs_100x['multi_pan_table'][4]
    assert pan5['num_pans'] == 5
    assert np.isclose(pan5['prob_at_least_one_pct'], 4.90, atol=0.1)
    assert np.isclose(pan5['expected_lots'], 0.05, atol=0.001)

    # Moderate subscription (10x)
    probs_10x = calculate_allotment_probabilities(10.0, max_pans=3)
    assert probs_10x['p_retail_single_pct'] == 10.0
    # 3 PANs: 1 - (1 - 0.10)^3 = 1 - 0.729 = 0.271 (27.1%)
    pan3 = probs_10x['multi_pan_table'][2]
    assert np.isclose(pan3['prob_at_least_one_pct'], 27.1, atol=0.1)

    # Undersubscribed (0.8x)
    probs_under = calculate_allotment_probabilities(0.8)
    assert probs_under['p_retail_single_pct'] == 100.0
    assert probs_under['p_retail_ratio'] == "Guaranteed Allotment (100%)"


def test_capital_allocation_rules():
    """Verify capital sizing, ASBA costs, and recommendation logic."""
    # Case 1: Insufficient capital (< 1 lot)
    insuf = calculate_capital_allocation(
        available_capital_inr=50000.0,
        issue_price=100.0,
        lot_size=1200,
        risk_quintile='Q1',
        retail_subs_times=25.0
    )
    assert insuf['status'] == 'INSUFFICIENT_CAPITAL'
    assert insuf['shortfall_inr'] == 70000.0

    # Case 2: Q1 High Conviction with multiple PANs and capital for 3 lots
    alloc_q1 = calculate_capital_allocation(
        available_capital_inr=400000.0,
        issue_price=100.0,
        lot_size=1200,  # 1.2 Lakhs per lot
        risk_quintile='Q1',
        retail_subs_times=20.0,
        family_pans=3
    )
    assert alloc_q1['status'] == 'SUCCESS'
    assert alloc_q1['recommended_lots'] == 3
    assert alloc_q1['recommended_capital_inr'] == 360000.0
    assert alloc_q1['spare_capital_inr'] == 40000.0
    assert "STRONG SUBSCRIBE" in alloc_q1['recommended_action']
    assert alloc_q1['conviction_score'] >= 90
    assert alloc_q1['total_asba_opportunity_cost_inr'] > 0

    # Case 3: Q5 Extreme Hazard / Value Trap -> Should recommend 0 lots and AVOID
    alloc_q5 = calculate_capital_allocation(
        available_capital_inr=500000.0,
        issue_price=100.0,
        lot_size=1200,
        risk_quintile='Q5',
        retail_subs_times=15.0,
        family_pans=3
    )
    assert alloc_q5['recommended_lots'] == 0
    assert alloc_q5['recommended_capital_inr'] == 0.0
    assert "AVOID" in alloc_q5['recommended_action']


def test_end_to_end_recommendation_engine():
    """Verify full pipeline output structure."""
    engine = SMERecommendationEngine()
    rec = engine.generate_recommendation(
        company_name="Monolith Precision Ltd",
        issue_price=143.0,
        retail_subs_times=45.0,
        total_subs_times=85.0,
        available_capital_inr=300000.0,
        family_pans=2
    )
    assert rec['company_name'] == "Monolith Precision Ltd"
    assert rec['lot_size'] == 1000
    assert rec['min_amount_inr'] == 143000.0
    assert 'risk_profile' in rec
    assert 'allotment_info' in rec
    assert 'allocation_info' in rec
    assert rec['allocation_info']['recommended_lots'] in [0, 1, 2]


def test_prospectus_scenarios():
    """Verify that what-if scenario matrix calculates accurate binomial lottery odds."""
    from src.models.recommendation_engine import get_prospectus_scenarios
    df_scen = get_prospectus_scenarios(user_pans=3)
    assert len(df_scen) == 6
    assert 'Demand Scenario' in df_scen.columns
    assert 'Single PAN Odds' in df_scen.columns
    assert 'Combined Odds (3 PANs)' in df_scen.columns


def test_prospectus_fundamentals_evaluation():
    """Verify pure fundamental appraisal before any bidding data exists."""
    from src.models.recommendation_engine import evaluate_prospectus_fundamentals
    # Sound fundamental company
    eval_good = evaluate_prospectus_fundamentals(
        issue_price=100.0,
        lot_size=1200,
        firm_age=15.0,
        pe_ratio=18.0,
        debt_to_asset=0.25,
        available_capital_inr=500000.0,
        family_pans=3
    )
    assert eval_good['composite_fundamental_score'] >= 75
    assert "STRONG" in eval_good['verdict']
    assert eval_good['recommended_lots'] == 3
    assert eval_good['min_amount_inr'] == 120000.0

    # Overpriced debt-ridden company
    eval_bad = evaluate_prospectus_fundamentals(
        issue_price=100.0,
        lot_size=1200,
        firm_age=2.0,
        pe_ratio=85.0,
        debt_to_asset=0.85,
        available_capital_inr=500000.0,
        family_pans=3
    )
    assert eval_bad['composite_fundamental_score'] < 50
    assert "TRAP" in eval_bad['verdict']
    assert eval_bad['recommended_lots'] == 0

