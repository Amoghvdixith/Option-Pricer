"""Unified API: method/style safeguards, agreement across methods, doctest example."""

import doctest

import pytest

import optpricer.api
from optpricer import MarketData, Option, UnsupportedMethodError, price, price_all, supported


def test_unsupported_combinations_raise():
    am = Option("put", 100, 1.0, "american")
    m = MarketData(100, 0.05, 0.0, 0.2)
    for meth in ("bs", "mc"):
        assert not supported(meth, "american")
        with pytest.raises(UnsupportedMethodError):
            price(am, m, meth)
    with pytest.raises(UnsupportedMethodError):
        price(am, m, "heston")


def test_price_all_lists_unsupported_with_reason():
    df = price_all(Option("put", 100, 1.0, "american"), MarketData(100, 0.05, 0.0, 0.2),
                   methods=["bs", "lr", "mc"], settings={"lr": {"n": 101}})
    st = df.set_index("method")["status"]
    assert st["bs"].startswith("unsupported") and st["mc"].startswith("unsupported") and st["lr"] == "ok"


def test_european_methods_agree(base_market):
    o = Option("call", 100, 1.0)
    df = price_all(o, base_market, settings={"mc": {"n_paths": 200_000, "seed": 1}, "explicit": {"n_time": 4000}})
    ref = df.set_index("method").loc["bs", "price"]
    for _, row in df.iterrows():
        tol = 4 * row["std_error"] if row["method"] == "mc" else 3e-3
        assert abs(row["price"] - ref) < tol, row


def test_doctest_examples():
    res = doctest.testmod(optpricer.api)
    assert res.failed == 0


def test_cli_price_and_compare(capsys):
    from optpricer.cli import main

    assert main(["price", "--kind", "put", "--style", "american", "--method", "lr", "--n", "201"]) == 0
    out = capsys.readouterr().out
    assert '"method": "lr"' in out
    assert main(["compare", "--kind", "call"]) == 0
    assert main(["price", "--style", "american", "--method", "bs"]) == 2


def test_price_all_reports_unstable_explicit_instead_of_aborting(base_market):
    df = price_all(Option("call", 100, 1.0), base_market, methods=["bs", "explicit"],
                   settings={"explicit": {"n_space": 400, "n_time": 50}})
    st = df.set_index("method")["status"]
    assert st["bs"] == "ok" and "UnstableSchemeError" in st["explicit"]


def test_config_and_paths():
    from optpricer.paths import load_config, project_root

    root = project_root()
    assert (root / "pyproject.toml").exists()
    fast, research = load_config("hedging", "fast"), load_config("hedging", "research")
    assert fast["n_paths"] < research["n_paths"] and fast["seed"] == research["seed"]
    with pytest.raises(ValueError):
        load_config("hedging", "turbo")
