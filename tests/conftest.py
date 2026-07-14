"""Shared test setup.

`pricing.py` imports Streamlit (for its `@st.cache_data` decorator). The pure
functions under test don't need a Streamlit runtime, so when Streamlit isn't
installed we register a minimal stub whose `cache_data` is a pass-through
decorator. When the real Streamlit *is* installed (the app's own environment),
we leave it untouched.
"""

import sys
import types


def _install_streamlit_stub() -> None:
    try:
        import streamlit  # noqa: F401

        return  # real Streamlit present — nothing to do
    except ModuleNotFoundError:
        pass

    stub = types.ModuleType("streamlit")

    def cache_data(*args, **kwargs):
        # Support both @st.cache_data and @st.cache_data(ttl=...) usages.
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator

    stub.cache_data = cache_data
    sys.modules["streamlit"] = stub


_install_streamlit_stub()
