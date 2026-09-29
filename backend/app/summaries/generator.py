"""
Plain-language summaries for OutbreakWatch.

Per the concept note: "an automatically generated plain-language summary
describes the situation for staff, constrained so that it explains the
numbers but never invents them."

This is deterministic templating, not an LLM. Every number and claim in the
output text is read directly from a value already computed by the
forecasting, anomaly_detection, or resource_tracking modules — there is no
generation step that could hallucinate a fact. If we later want an LLM to
smooth the phrasing, it should be given these already-computed facts as
fixed inputs to rephrase, never allowed to introduce new figures.
"""

from dataclasses import dataclass
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../anomaly_detection"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../resource_tracking"))
from detector import AnomalyFlag
from tracker import StockoutPrediction


def summarize_condition(
    facility: str,
    condition: str,
    recent_series: pd.Series,
    forecast_df: pd.DataFrame,
    recent_flags: list[AnomalyFlag],
    lookback_days: int = 7,
) -> str:
    """
    recent_series: the observed case-count series (date-indexed).
    forecast_df: output of forecaster.forecast() — mean/lower/upper by date.
    recent_flags: AnomalyFlag list already filtered to relevant recent dates
        (e.g. last `lookback_days`), from detector.detect_all().
    """
    last_actual = recent_series.iloc[-lookback_days:].sum()
    last_date = recent_series.index[-1]

    next_period_forecast = forecast_df.iloc[:lookback_days]["mean"].sum()
    next_period_lower = forecast_df.iloc[:lookback_days]["lower"].sum()
    next_period_upper = forecast_df.iloc[:lookback_days]["upper"].sum()

    prior_period = recent_series.iloc[-2 * lookback_days:-lookback_days].sum()
    if prior_period > 0:
        pct_change = (last_actual - prior_period) / prior_period * 100
        trend_phrase = (
            f"up {pct_change:.0f}% from the {lookback_days} days before that"
            if pct_change > 5 else
            f"down {abs(pct_change):.0f}% from the {lookback_days} days before that"
            if pct_change < -5 else
            "roughly flat compared to the prior period"
        )
    else:
        trend_phrase = "no comparable prior-period data"

    lines = [
        f"{facility} — {condition}: {int(last_actual)} cases in the last {lookback_days} days "
        f"(through {last_date.date()}), {trend_phrase}.",
        f"Next {lookback_days} days are forecast at {next_period_forecast:.0f} cases "
        f"(range {next_period_lower:.0f}-{next_period_upper:.0f}, 95% interval).",
    ]

    active_alerts = [f for f in recent_flags if f.severity == "alert"]
    active_watches = [f for f in recent_flags if f.severity == "watch"]

    if active_alerts:
        dates = ", ".join(str(f.date.date()) for f in sorted(set(f.date for f in active_alerts)))
        lines.append(
            f"ALERT: case counts on {dates} were significantly higher than the model expected "
            f"for this time of year — recommend review."
        )
    elif active_watches:
        dates = ", ".join(str(f.date.date()) for f in sorted(set(f.date for f in active_watches)))
        lines.append(f"Watch: elevated-but-not-critical deviation flagged on {dates}.")
    else:
        lines.append("No anomaly flags in the recent period.")

    return "\n".join(lines)


def summarize_resource(prediction: StockoutPrediction) -> str:
    """
    Builds a plain-language line from an already-computed StockoutPrediction.
    Does not recompute anything — only phrases what's already in the object.
    """
    header = (
        f"{prediction.facility} — {prediction.resource}: "
        f"current level {prediction.current_level:.0f} "
        f"(safety threshold {prediction.safety_threshold:.0f})."
    )

    if prediction.days_until_stockout_mean is None:
        return header + f" No stock-out projected within the {prediction.forecast_horizon_days}-day horizon."

    bounds = [d for d in (prediction.days_until_stockout_optimistic, prediction.days_until_stockout_pessimistic) if d is not None]
    range_str = f"{min(bounds)}-{max(bounds)}" if bounds else "unknown"

    return (
        header + f" At current consumption trends, projected to reach the safety threshold in "
        f"{prediction.days_until_stockout_mean} days "
        f"(range: {range_str} days depending on how consumption trends). Recommend scheduling resupply."
    )


def facility_situation_report(facility: str, condition_summaries: list[str], resource_summaries: list[str]) -> str:
    """Combines individual summaries into one situation report for a facility."""
    parts = [f"=== Situation Report: {facility} ===\n"]
    parts.append("-- Disease surveillance --")
    parts.extend(condition_summaries)
    parts.append("\n-- Resource status --")
    parts.extend(resource_summaries)
    return "\n".join(parts)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forecasting"))
    from loader import load_surveillance_csv, load_resource_csv
    from forecaster import forecast
    from detector import detect_all
    from tracker import predict_stockout

    data_dir = os.path.join(os.path.dirname(__file__), "../../data")
    surv_df, _ = load_surveillance_csv(os.path.join(data_dir, "surveillance_sample.csv"))
    res_df, _ = load_resource_csv(os.path.join(data_dir, "resources_sample.csv"))

    facility = "Kisumu Referral Hospital"

    sub = surv_df[(surv_df.facility == facility) & (surv_df.condition == "Malaria")]
    series = sub.set_index("date")["case_count"]
    fc = forecast(series, steps=14)
    flags = detect_all(series)
    recent_flags = [f for f in flags if f.date >= series.index[-1] - pd.Timedelta(days=14)]
    condition_summary = summarize_condition(facility, "Malaria", series, fc, recent_flags)

    res_sub = res_df[(res_df.facility == facility) & (res_df.resource == "ORS Sachets")]
    res_series = res_sub.set_index("date")["level"].loc[:"2023-04-18"]
    stockout_pred = predict_stockout(res_series, facility, "ORS Sachets", forecast_horizon_days=30)
    resource_summary = summarize_resource(stockout_pred)

    report = facility_situation_report(facility, [condition_summary], [resource_summary])
    print(report)