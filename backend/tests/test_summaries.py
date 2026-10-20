import pandas as pd
from generator import summarize_condition, summarize_resource
from detector import AnomalyFlag
from tracker import StockoutPrediction


def _prediction(kind="consumable", mean=17, best=24, worst=13):
    return StockoutPrediction(
        facility="Kisumu Referral Hospital", resource="Thing",
        current_level=3780, safety_threshold=500,
        days_until_stockout_mean=mean,
        days_until_stockout_optimistic=best,
        days_until_stockout_pessimistic=worst,
        forecast_horizon_days=30, kind=kind,
    )


def test_consumable_at_risk_wording():
    text = summarize_resource(_prediction())
    assert "17 days" in text
    assert "13-24" in text                 # range always reads low-to-high
    assert "resupply" in text


def test_capacity_wording_differs():
    text = summarize_resource(_prediction(kind="capacity", mean=20, best=30, worst=12))
    assert "current availability" in text
    assert "surge" in text
    assert "resupply" not in text


def test_no_risk_message():
    text = summarize_resource(_prediction(mean=None, best=None, worst=None))
    assert "No stock-out projected within the 30-day horizon" in text


def test_worst_case_only():
    text = summarize_resource(_prediction(mean=None, best=None, worst=25))
    assert "worst case" in text
    assert "25 days" in text


def _case_inputs():
    dates = pd.date_range("2026-08-04", periods=28, freq="D")          # ends 2026-08-31
    series = pd.Series([10] * 21 + [20] * 7, index=dates)
    future = pd.date_range("2026-09-01", periods=7, freq="D")
    fc = pd.DataFrame({"mean": 20.0, "lower": 10.0, "upper": 30.0}, index=future)
    return series, fc


def test_condition_summary_numbers_come_from_the_inputs():
    series, fc = _case_inputs()
    text = summarize_condition("Facility", "Malaria", series, fc, [])

    assert "140 cases in the last 7 days" in text
    assert "up 100%" in text
    assert "forecast at 140 cases" in text
    assert "70-210" in text
    assert "No anomaly flags" in text


def test_condition_summary_reports_alerts():
    series, fc = _case_inputs()
    flag = AnomalyFlag(pd.Timestamp("2026-08-30"), "residual_zscore", 20, 10.0, 5.0, "alert")
    text = summarize_condition("Facility", "Malaria", series, fc, [flag])
    assert "ALERT" in text
    assert "2026-08-30" in text