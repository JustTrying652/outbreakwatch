import pandas as pd
import pytest
from tracker import predict_stockout, resource_kind, _daily_consumption
from ground_truth import STOCKOUT_FACILITY, STOCKOUT_RESOURCE, REPLAY_DATE


def _daily(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


# ---------- fast: no model fitting ----------

def test_resource_kind():
    assert resource_kind("ICU Beds") == "capacity"
    assert resource_kind("ORS Sachets") == "consumable"


def test_resupply_jump_is_not_negative_consumption():
    consumption = _daily_consumption(_daily([100, 90, 80, 500, 490]))
    assert list(consumption) == [10, 10, 0, 10]


def test_consumable_fallback_on_short_history():
    levels = [1000 - 10 * d for d in range(31)]      # steady 10/day, ends at 700
    result = predict_stockout(_daily(levels), "A", "Medicine", safety_threshold=100, forecast_horizon_days=90)
    assert result.kind == "consumable"
    assert result.days_until_stockout_mean == 60      # (700 - 100) / 10


def test_capacity_fallback_flags_only_when_already_at_threshold():
    healthy = predict_stockout(_daily([40] * 30), "A", "ICU Beds", safety_threshold=4)
    assert healthy.kind == "capacity"
    assert healthy.days_until_stockout_mean is None

    critical = predict_stockout(_daily([3] * 30), "A", "ICU Beds", safety_threshold=4)
    assert critical.days_until_stockout_mean == 1


# ---------- slow: real data, fits ETS models ----------

@pytest.mark.slow
def test_replayed_stockout_is_predicted_with_lead_time(level_series):
    series = level_series(STOCKOUT_FACILITY, STOCKOUT_RESOURCE).loc[:REPLAY_DATE]
    r = predict_stockout(series, STOCKOUT_FACILITY, STOCKOUT_RESOURCE, forecast_horizon_days=30)

    assert r.kind == "consumable"
    assert r.days_until_stockout_mean is not None
    assert 10 <= r.days_until_stockout_mean <= 25                    # observed 17
    assert r.days_until_stockout_pessimistic <= r.days_until_stockout_mean
    assert r.days_until_stockout_mean <= r.days_until_stockout_optimistic


@pytest.mark.slow
def test_no_stockout_projected_for_latest_data(level_series):
    r = predict_stockout(level_series(STOCKOUT_FACILITY, STOCKOUT_RESOURCE),
                         STOCKOUT_FACILITY, STOCKOUT_RESOURCE, forecast_horizon_days=30)
    assert r.days_until_stockout_mean is None
    assert r.days_until_stockout_optimistic is None
    assert r.days_until_stockout_pessimistic is None


@pytest.mark.slow
def test_icu_beds_use_the_capacity_model_and_report_no_shortfall(level_series):
    """Regression test for the bed-modelling bug: beds were once treated as a consumable."""
    r = predict_stockout(level_series(STOCKOUT_FACILITY, "ICU Beds"),
                         STOCKOUT_FACILITY, "ICU Beds", forecast_horizon_days=90)
    assert r.kind == "capacity"
    assert r.days_until_stockout_mean is None