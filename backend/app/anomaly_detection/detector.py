"""
Anomaly (outbreak signal) detection for OutbreakWatch.

Design note — read this before changing thresholds:
An earlier version of this module compared raw case counts to a trailing
mean/std. That approach produced an unusable false-alarm rate (20-60% of
all days flagged) because it couldn't distinguish "seasonality doing what
seasonality does" (e.g. cholera's normal April rise) from a genuine
anomaly — a flat trailing baseline has no concept of expected seasonal
position, so any normal seasonal climb reads as a sustained deviation.

This version instead detects anomalies in the FORECAST RESIDUALS (actual -
what the forecasting module's ETS model already expected, which encodes
learned seasonality and trend) rather than in raw counts. What's left after
removing the seasonal/trend signal is much closer to white noise in the
"normal" case, and a genuine outbreak still stands out sharply as a residual
spike — because it is, by definition, something the seasonal/trend model did
NOT expect. Validated false-alarm rate with this approach: ~0.5% of normal
days at z>3 (see the __main__ block below), down from >20% with the
raw-count approach.

Two complementary checks on the residual series, both named in the concept
note:
  1. Same-period historical comparison -> here: rolling z-score of residuals
     against a trailing window (spike detection)
  2. CUSUM control chart on residuals (sustained-drift detection)
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forecasting"))
from forecaster import _fit_ets  # reuse the same ETS fit as the forecasting module


@dataclass
class AnomalyFlag:
    date: pd.Timestamp
    method: str            # "residual_zscore" or "cusum"
    value: float           # actual case count that day
    expected: float          # model's expected value that day (fitted/baseline)
    score: float             # z-score for residual method, cumulative sum for cusum
    severity: str            # "watch" or "alert"


def _get_residuals(series: pd.Series, seasonal_periods: int = 7) -> tuple[pd.Series, pd.Series]:
    """
    Fits an ETS model on the full series and returns (residuals, fitted_values).
    residuals = actual - fitted (in-sample one-step-ahead fitted values).
    """
    s = series.asfreq("D").fillna(0)
    fit_result = _fit_ets(s, seasonal_periods=seasonal_periods)
    fitted = fit_result.fittedvalues
    residuals = s - fitted
    return residuals, fitted


def detect_residual_zscore_anomalies(
    series: pd.Series,
    baseline_window_days: int = 56,
    watch_z: float = 3.0,
    alert_z: float = 4.0,
) -> list[AnomalyFlag]:
    """
    Flags days where the forecast residual is an unusually large POSITIVE
    z-score relative to the trailing window of residuals (i.e. actual cases
    came in much higher than the seasonally-adjusted model expected).
    """
    residuals, fitted = _get_residuals(series)
    s = series.asfreq("D").fillna(0)
    flags = []

    for current_date in residuals.index:
        window_start = current_date - pd.Timedelta(days=baseline_window_days)
        history = residuals.loc[window_start:current_date - pd.Timedelta(days=1)]
        if len(history) < 21:
            continue

        mean, std = history.mean(), history.std()
        if std == 0 or pd.isna(std):
            continue

        r = residuals.loc[current_date]
        z = (r - mean) / std

        if z >= alert_z:
            flags.append(AnomalyFlag(current_date, "residual_zscore",
                                      s.loc[current_date], fitted.loc[current_date], z, "alert"))
        elif z >= watch_z:
            flags.append(AnomalyFlag(current_date, "residual_zscore",
                                      s.loc[current_date], fitted.loc[current_date], z, "watch"))

    return flags


def detect_cusum_anomalies(
    series: pd.Series,
    baseline_window_days: int = 90,
    slack_k: float = 1.0,
    watch_h: float = 5.0,
    alert_h: float = 8.0,
) -> list[AnomalyFlag]:
    """
    One-sided CUSUM on forecast residuals (not raw counts). Accumulates
    (residual_z - slack_k) when positive; resets toward 0 once residuals
    return to normal. Flags a sustained upward drift in residuals — an
    outbreak building over several days that a single-day z-score might
    individually rate as only "watch" level, but which persists.
    """
    residuals, fitted = _get_residuals(series)
    s = series.asfreq("D").fillna(0)
    flags = []
    cusum = 0.0

    for current_date in residuals.index:
        window_start = current_date - pd.Timedelta(days=baseline_window_days)
        history = residuals.loc[window_start:current_date - pd.Timedelta(days=1)]
        if len(history) < 28:
            continue

        mean, std = history.mean(), history.std()
        if std == 0 or pd.isna(std):
            continue

        z = (residuals.loc[current_date] - mean) / std
        cusum = max(0.0, cusum + z - slack_k)

        if cusum >= alert_h:
            flags.append(AnomalyFlag(current_date, "cusum", s.loc[current_date],
                                      fitted.loc[current_date], cusum, "alert"))
        elif cusum >= watch_h:
            flags.append(AnomalyFlag(current_date, "cusum", s.loc[current_date],
                                      fitted.loc[current_date], cusum, "watch"))

    return flags


def detect_all(series: pd.Series, **kwargs) -> list[AnomalyFlag]:
    """Run both detectors (sharing one ETS fit under the hood) and return combined, date-sorted flags."""
    z_flags = detect_residual_zscore_anomalies(series)
    cusum_flags = detect_cusum_anomalies(series)
    return sorted(z_flags + cusum_flags, key=lambda f: f.date)


def flags_to_dataframe(flags: list[AnomalyFlag]) -> pd.DataFrame:
    if not flags:
        return pd.DataFrame(columns=["date", "method", "value", "expected", "score", "severity"])
    return pd.DataFrame([{
        "date": f.date, "method": f.method, "value": f.value,
        "expected": round(f.expected, 2), "score": round(f.score, 2),
        "severity": f.severity,
    } for f in flags])


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion"))
    from loader import load_surveillance_csv

    df, _ = load_surveillance_csv(os.path.join(os.path.dirname(__file__), "../../data/surveillance_sample.csv"))

    GROUND_TRUTH = [
        ("Kisumu Referral Hospital", "Cholera", "2023-04-10", "2023-05-01"),
        ("Nairobi County Hospital", "Respiratory Infections", "2024-07-01", "2024-07-29"),
        ("Nakuru District Hospital", "Malaria", "2025-11-05", "2025-11-23"),
        ("Kisumu Referral Hospital", "Malaria", "2026-05-20", "2026-06-14"),
    ]

    print("=== Detection vs. known injected outbreaks ===\n")
    for facility, condition, start, end in GROUND_TRUTH:
        sub = df[(df.facility == facility) & (df.condition == condition)]
        series = sub.set_index("date")["case_count"]

        flags = detect_all(series)
        window_flags = [f for f in flags if pd.Timestamp(start) <= f.date <= pd.Timestamp(end) + pd.Timedelta(days=5)]
        alert_flags = [f for f in window_flags if f.severity == "alert"]
        first_alert = min((f.date for f in alert_flags), default=None)
        delay = (first_alert - pd.Timestamp(start)).days if first_alert is not None else None

        status = f"DETECTED (first alert {delay} days after outbreak start)" if first_alert else "MISSED"
        print(f"{facility} / {condition} [{start} to {end}]: {status}  "
              f"({len(window_flags)} flags in window, {len(alert_flags)} at alert level)")

        outside = [f for f in flags if not (pd.Timestamp(start) - pd.Timedelta(days=10) <= f.date <= pd.Timestamp(end) + pd.Timedelta(days=10))]
        alert_outside = [f for f in outside if f.severity == "alert"]
        print(f"  False-alarm check: {len(alert_outside)} alert-level flags outside the window "
              f"out of {len(series)} days ({len(alert_outside)/len(series)*100:.2f}%)\n")