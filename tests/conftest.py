"""Shared fixtures. All tests are offline and seeded."""

from __future__ import annotations

import pytest

from optpricer import MarketData, Option


@pytest.fixture
def base_market() -> MarketData:
    return MarketData(S=100.0, r=0.05, q=0.0, sigma=0.20)


@pytest.fixture
def atm_call() -> Option:
    return Option("call", 100.0, 1.0)


@pytest.fixture
def atm_put() -> Option:
    return Option("put", 100.0, 1.0)
