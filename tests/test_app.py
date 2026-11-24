"""Streamlit app smoke tests (offline). Uses streamlit.testing (no browser)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytestmark = pytest.mark.app


def test_app_loads_without_exceptions():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_pricing_button_european_and_american():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()  # "Price" with defaults (European call)
    assert not at.exception
    assert len(at.dataframe) >= 1
    at.radio[1].set_value("american").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception


def test_synthetic_smile_and_hedging_buttons():
    at = AppTest.from_file(APP, default_timeout=180).run()
    labels = [b.label for b in at.button]
    at.button[labels.index("Build synthetic smiles")].click().run()
    assert not at.exception
    at.button[labels.index("Run hedging simulation")].click().run()
    assert not at.exception
