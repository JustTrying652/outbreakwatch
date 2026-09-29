"""
Resource tracking & stock-out prediction for OutbreakWatch.

Per the concept note: "Resource levels are treated as time-series in their
own right, so the same forecasting engine — applied in the depletion
direction — predicts when a medicine supply or bed capacity will fall below
a safe threshold, ahead of time."

Design note: raw resource LEVEL data isn't smooth depletion — it's a
sawtooth (steady decline, then a jump back up on resupply). Forecasting the
raw level directly would have the model try to learn resupply timing, which
isn't the point. Instead:
  1. Convert level -> daily CONSUMPTION (previous - current, clipped at 0 so
     a resupply jump reads as "0 consumed that day", not negative consumption)
  2. Forecast consumption forward using the same ETS engine as the
     forecasting module
  3. Simulate the stock trajectory forward from the current level, assuming
     NO further resupply, and find the day it crosses a safety threshold

That last assumption is deliberate, not an oversight: "when would we run out
if nothing changes" is exactly the early-warning question a resupply
decision-maker needs answered, well before the shelf is actually empty.
Three trajectories (mean / low-consumption / high-consumption bound) are
reported so the uncertainty is visible, not just a single point guess.
"""

from dataclasses import dataclass
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forecasting"))
from forecaster import _fit_ets


@dataclass
class StockoutPrediction:
    facility: str
    resource: str
    current_level: float
    safety_threshold: float
    days_until_stockout_mean: int | None
    days_until_stockout_optimistic: int | None
    days_until_stockout_pessimistic: int | None
    forecast_horizon_days: int

    def summary(self) -> str:
        def fmt(d):
            return f"{d} days" if d is not None else f"not within {self.forecast_horizon_days}-day horizon"
        return (
            f"{self.facility} / {self.resource}: current level {self.current_level:.0f}, "
            f"safety threshold {self.safety_threshold:.0f}\n"
            f"  Best case:  stock-out in {fmt(self.days_until_stockout_optimistic)}\n"
            f"  Expected:   stock-out in {fmt(self.days_until_stockout_mean)}\n"
            f"  Worst case: stock-out in {fmt(self.days_until_stockout_pessimistic)}"
        )


def _daily_consumption(level_series: pd.Series) -> pd.Series:
    """previous_level - current_level, clipped at 0 (resupply days read as 0 consumed)."""
    s = level_series.asfreq("D").ffill()
    consumption = (s.shift(1) - s).clip(lower=0)
    return consumption.dropna()


def predict_stockout(
    level_series: pd.Series,
    facility: str,
    resource: str,
    safety_threshold: float | None = None,
    forecast_horizon_days: int = 90,
    seasonal_periods: int = 7,
) -> StockoutPrediction:
    """
    level_series: date-indexed resource level history for one facility/resource.
    safety_threshold: absolute level considered "critical". Defaults to 10% of
        the observed 90-day max level if not provided.
    """
    s = level_series.asfreq("D").ffill()
    current_level = float(s.iloc[-1])

    if safety_threshold is None:
        safety_threshold = 0.10 * s.tail(90).max()

    consumption = _daily_consumption(s)

    if len(consumption) < 60 or consumption.std() == 0:
        avg = max(consumption.mean(), 0.01)
        days_to_stockout = int((current_level - safety_threshold) / avg) if avg > 0 else None
        days_to_stockout = days_to_stockout if days_to_stockout and days_to_stockout >= 0 else None
        return StockoutPrediction(facility, resource, current_level, safety_threshold,
                                   days_to_stockout, days_to_stockout, days_to_stockout,
                                   forecast_horizon_days)

    fit_result = _fit_ets(consumption, seasonal_periods=seasonal_periods)
    pred = fit_result.get_prediction(start=len(consumption), end=len(consumption) + forecast_horizon_days - 1)
    frame = pred.summary_frame(alpha=0.05)

    mean_consumption = frame["mean"].clip(lower=0).values
    low_consumption = frame["pi_lower"].clip(lower=0).values
    high_consumption = frame["pi_upper"].clip(lower=0).values

    def days_to_cross(consumption_path):
        cumulative = np.cumsum(consumption_path)
        remaining = current_level - cumulative
        crossing = np.where(remaining <= safety_threshold)[0]
        return int(crossing[0]) + 1 if len(crossing) > 0 else None

    return StockoutPrediction(
        facility=facility, resource=resource,
        current_level=current_level, safety_threshold=safety_threshold,
        days_until_stockout_mean=days_to_cross(mean_consumption),
        days_until_stockout_optimistic=days_to_cross(low_consumption),
        days_until_stockout_pessimistic=days_to_cross(high_consumption),
        forecast_horizon_days=forecast_horizon_days,
    )


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion"))
    from loader import load_resource_csv

    df, _ = load_resource_csv(os.path.join(os.path.dirname(__file__), "../../data/resources_sample.csv"))

    sub = df[(df.facility == "Kisumu Referral Hospital") & (df.resource == "ORS Sachets")]
    full_series = sub.set_index("date")["level"]

    cutoff = pd.Timestamp("2023-04-18")  # a few days into the known suppression window (04-10 to 04-27)
    truncated_series = full_series.loc[:cutoff]

    print(f"Simulating as of {cutoff.date()} (partway through the known suppressed-resupply window)\n")
    result = predict_stockout(truncated_series, "Kisumu Referral Hospital", "ORS Sachets",
                               forecast_horizon_days=30)
    print(result.summary())

    print("\n--- For comparison, current real-world state (using full history) ---")
    result_now = predict_stockout(full_series, "Kisumu Referral Hospital", "ORS Sachets",
                                   forecast_horizon_days=30)
    print(result_now.summary())