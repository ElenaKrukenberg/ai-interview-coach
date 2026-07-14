"""OpenRouter model pricing lookup and per-request cost estimation.

Pricing is fetched live from the OpenRouter models endpoint (per-token USD
prices) and cached for an hour so the app does not refetch on every rerun.
"""

from typing import Dict, Optional

import requests
import streamlit as st

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
EXCHANGE_RATE_URL = "https://api.frankfurter.app/latest"
FALLBACK_USD_TO_EUR = 0.90  # used when the exchange-rate API is unreachable


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_models_catalog() -> list:
    """
    Fetch the raw OpenRouter models catalog (pricing, supported parameters).

    Returns an empty list if the endpoint is unreachable (catalog data is
    optional and must never break the app).
    """
    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except (requests.RequestException, ValueError):
        return []


def fetch_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    Per-token USD pricing for all OpenRouter models, derived from the catalog.

    Returns a mapping like {"openai/gpt-5-mini": {"prompt": 2.5e-07, "completion": 2e-06}}.
    """
    pricing = {}
    for model in _fetch_models_catalog():
        model_pricing = model.get("pricing", {})
        try:
            pricing[model["id"]] = {
                "prompt": float(model_pricing.get("prompt", 0)),
                "completion": float(model_pricing.get("completion", 0)),
            }
        except (TypeError, ValueError, KeyError):
            continue
    return pricing


def model_supports_parameter(model: str, parameter: str) -> bool:
    """
    Whether an OpenRouter model accepts a given API parameter — e.g. GPT-5
    reasoning models ignore `temperature` (OpenRouter drops it silently).

    Fails open: if the catalog is unreachable or the model is not listed, the
    parameter is assumed supported so the UI never locks up on a network hiccup.
    """
    for entry in _fetch_models_catalog():
        if entry.get("id") == model:
            supported = entry.get("supported_parameters")
            if not isinstance(supported, list):
                return True
            return parameter in supported
    return True


def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Optional[float]:
    """Estimate the USD cost of one request; None if pricing is unavailable."""
    model_pricing = fetch_model_pricing().get(model)
    if not model_pricing:
        return None
    return (
        prompt_tokens * model_pricing["prompt"]
        + completion_tokens * model_pricing["completion"]
    )


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_usd_to_eur() -> float:
    """Fetch the daily USD→EUR rate (ECB data); falls back to a fixed rate."""
    try:
        response = requests.get(
            EXCHANGE_RATE_URL, params={"from": "USD", "to": "EUR"}, timeout=10
        )
        response.raise_for_status()
        return float(response.json()["rates"]["EUR"])
    except (requests.RequestException, ValueError, KeyError):
        return FALLBACK_USD_TO_EUR


def format_cost(cost_usd: Optional[float]) -> str:
    """
    Format a USD cost as euros, e.g. '€0,0012'.

    None means "cost unknown" (no request yet, or pricing unavailable) and
    renders as 'n/a' — distinct from a true zero. Micro-amounts get extra
    decimals so they never collapse into an unreadable string of zeros.
    """
    if cost_usd is None:
        return "n/a"
    if not cost_usd:
        return "€0,00"

    eur = cost_usd * fetch_usd_to_eur()
    if eur < 0.0001:
        text = f"{eur:.6f}"
    elif eur < 0.01:
        text = f"{eur:.4f}"
    else:
        text = f"{eur:.2f}"
    return "€" + text.replace(".", ",")
