"""Unit tests for cost estimation and EUR formatting.

Network-touching helpers (`fetch_model_pricing`, `fetch_usd_to_eur`) are
monkeypatched so tests are offline and deterministic.
"""

import pytest

import pricing
from pricing import estimate_cost, format_cost


@pytest.fixture(autouse=True)
def fixed_rate(monkeypatch):
    """Pin USD->EUR to 0.5 so expected euro values are trivial to compute."""
    monkeypatch.setattr(pricing, "fetch_usd_to_eur", lambda: 0.5)


@pytest.fixture
def fake_pricing(monkeypatch):
    monkeypatch.setattr(
        pricing,
        "fetch_model_pricing",
        lambda: {"openai/gpt-5-mini": {"prompt": 1e-06, "completion": 2e-06}},
    )


class TestEstimateCost:
    def test_basic_computation(self, fake_pricing):
        # 1000 * 1e-6 + 500 * 2e-6 = 0.001 + 0.001 = 0.002 USD
        cost = estimate_cost("openai/gpt-5-mini", 1000, 500)
        assert cost == pytest.approx(0.002)

    def test_zero_tokens_is_zero(self, fake_pricing):
        assert estimate_cost("openai/gpt-5-mini", 0, 0) == 0.0

    def test_unknown_model_returns_none(self, fake_pricing):
        assert estimate_cost("made-up/model", 1000, 500) is None

    def test_no_pricing_available_returns_none(self, monkeypatch):
        monkeypatch.setattr(pricing, "fetch_model_pricing", lambda: {})
        assert estimate_cost("openai/gpt-5-mini", 1000, 500) is None


class TestFormatCost:
    def test_none_renders_not_available(self):
        # Regression: None (unknown) must be distinct from a true zero.
        assert format_cost(None) == "n/a"

    def test_zero_renders_euro_zero(self):
        assert format_cost(0.0) == "€0,00"

    def test_uses_comma_decimal_separator(self):
        assert "," in format_cost(0.02)
        assert "." not in format_cost(0.02)

    def test_normal_amount_two_decimals(self):
        # 0.04 USD * 0.5 = 0.02 EUR
        assert format_cost(0.04) == "€0,02"

    def test_small_amount_four_decimals(self):
        # 0.002 USD * 0.5 = 0.001 EUR -> < 0.01 branch, 4 decimals
        assert format_cost(0.002) == "€0,0010"

    def test_micro_amount_six_decimals(self):
        # 0.0001 USD * 0.5 = 0.00005 EUR -> < 0.0001 branch, 6 decimals
        assert format_cost(0.0001) == "€0,000050"
