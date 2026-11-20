from test_risk import LIMITS, account, decide


def test_registered_risk_worker_fits_margin_after_full_sizing():
    settings = dict(
        adaptive_={"outcomes": []},
        limits_={**LIMITS, "max_leverage": 30.0},
        venue_max_leverage=75,
    )
    ample = decide(**settings)
    first = ample["evaluation"]["decision"]
    assert first["approved"] and first["leverage"] == 1
    # Force a 2x requirement without changing the exposure, the risk budget or the market.
    available = first["notional_usd"] * 0.6
    tight = decide(**settings, account_=lambda: account(available_usd=available))
    decision = tight["evaluation"]["decision"]
    assert decision["approved"] and decision["leverage"] == 2
    assert tight["adaptive"]["leverage"] == "2"
    assert tight["adaptive"]["leverage_ceiling"] == "3"
    assert decision["quantity"] == first["quantity"]
    assert decision["worst_case_loss_usd"] == first["worst_case_loss_usd"]

    empty = decide(**settings, account_=lambda: account(available_usd=0))
    assert not empty["evaluation"]["decision"]["approved"]
    assert "position_capacity_exhausted" in empty["evaluation"]["decision"]["rejection_reasons"]
