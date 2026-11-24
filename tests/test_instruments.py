import datetime as dt
import math

import numpy as np
import pytest

from optpricer import InputError, MarketData, Option
from optpricer.conventions import DAYS_PER_YEAR, to_display_units, year_fraction


def test_option_normalises_and_is_frozen():
    o = Option("C", 100, 0.5, "American")
    assert o.kind == "call" and o.style == "american"
    with pytest.raises(AttributeError):
        o.K = 5  # frozen dataclass


@pytest.mark.parametrize("kwargs", [
    dict(kind="call", K=0.0, T=1.0), dict(kind="call", K=-1.0, T=1.0), dict(kind="call", K=100, T=-0.1),
    dict(kind="straddle", K=100, T=1.0), dict(kind="call", K=math.nan, T=1.0), dict(kind="call", K=100, T=math.inf),
    dict(kind="call", K=100, T=1.0, style="bermudan"),
])
def test_option_rejects_invalid(kwargs):
    with pytest.raises(InputError):
        Option(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(S=-1), dict(S=100, sigma=-0.1), dict(S=math.nan), dict(S=100, r=math.inf)])
def test_market_rejects_invalid(kwargs):
    with pytest.raises(InputError):
        MarketData(**kwargs)


def test_market_allows_negative_rates_and_zero_vol():
    m = MarketData(100, r=-0.01, q=-0.005, sigma=0.0)
    assert m.r == -0.01 and m.sigma == 0.0


def test_year_fraction_act365():
    assert year_fraction(dt.date(2026, 1, 1), dt.date(2027, 1, 1)) == pytest.approx(365 / DAYS_PER_YEAR)
    yf = year_fraction(dt.datetime(2026, 1, 1, 10), dt.datetime(2026, 1, 1, 16))
    assert yf == pytest.approx(0.25 / 365)
    with pytest.raises(ValueError):
        year_fraction(dt.date(2026, 2, 1), dt.date(2026, 1, 1))


def test_display_units_scaling():
    d = to_display_units({"vega": 40.0, "rho": 50.0, "theta": -3.65, "delta": 0.5, "volga": 1e4, "vanna": 2.0})
    assert d["vega"] == pytest.approx(0.40)
    assert d["rho"] == pytest.approx(0.50)
    assert d["theta"] == pytest.approx(-0.01)
    assert d["delta"] == 0.5
    assert d["volga"] == pytest.approx(1.0)
    assert d["vanna"] == pytest.approx(0.02)
    arr = to_display_units({"vega": np.array([100.0, 200.0])})["vega"]
    np.testing.assert_allclose(arr, [1.0, 2.0])
