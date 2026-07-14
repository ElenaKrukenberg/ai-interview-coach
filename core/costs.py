"""Cost accounting: turn API token usage into an estimated USD amount.

Pure business logic — session-state bookkeeping lives in core.state.
"""

from typing import Optional

from pricing import estimate_cost


def usage_cost(model: str, usage: Optional[object]) -> Optional[float]:
    """Estimated USD cost of one request from its token usage; None if unknown."""
    if usage is None:
        return None
    return estimate_cost(
        model=model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
