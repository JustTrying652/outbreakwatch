"""
Resource tracking & shortfall prediction for OutbreakWatch.

Per the concept note: "Resource levels are treated as time-series in their
own right, so the same forecasting engine ... predicts when a medicine supply
or bed capacity will fall below a safe threshold, ahead of time."

Two kinds of resource need two different models:

CONSUMABLE (medicines): raw level is a sawtooth (steady decline, then a jump
on resupply). Forecasting the level directly would make the model try to
learn resupply timing. Instead:
  1. Convert level -> daily CONSUMPTION (previous - current, clipped at 0 so
     a resupply jump reads as "0 consumed", not negative consumption)
  2. Forecast consumption with the same ETS engine as the forecasting module
  3. Simulate the stock forward from the current level assuming NO further
     resupply, and find the day it crosses the safety threshold
"If nothing changes, when do we run out" is the early-warning question a
resupply decision-maker needs answered.

CAPACITY (beds): beds are not consumed. They free up again when patients are
discharged. Clipping falling levels into "consumption" and ignoring rises
would only ever count the downward wiggles and wildly overstate depletion.
Instead the available level is forecast DIRECTLY with ETS, and we find the
day the forecast (with its prediction interval) reaches the safety threshold.

Either way, three trajectories (expected / best case / worst case) are
reported so the uncertainty is visible, not just a single point guess.
"""

from dataclasses import dataclass
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forecasting"))
from forecaster import _fit_ets

# Resources that are capacity rather than consumables. Add staff etc. here later.
CAPACITY_RESOURCES = {"ICU Beds"}


def resource_kind(resource: str) -> str:
    return "capacity" if resource in CAPACITY_RESOURCES else "consumable"


@dataclass
class StockoutPrediction:
    facility: str
    resource: str
    current_level: float
    safety_threshold: float
    days_until_stockout_mean: int | None        # None = doesn't cross threshold within horizon
    days_until_stockout_optimistic: int | None  # best case
    days_until_stockout_pessimistic: int | None # worst case
    forecast_horizon_days: int
    kind: str = "consumable"                    # "consumable" or "capacity"

    def summary(self) -> str:
        def fmt(d):
            return f"{d} days" if d is not None else f"not within {self.forecast_horizon_days}-day horizon"
        event = "shortfall" if self.kind == "capacity" else "stock-out"
        return (
            f"{self.facility} / {self.resource} [{self.kind}]: current level {self.current_level:.0f}, "
            f"safety threshold {self.safety_threshold:.0f}\n"
            f"  Best case:  {event} in {fmt(self.days_until_stockout_optimistic)}\n"
            f"  Expected:   {event} in {fmt(self.days_until_stockout_mean)}\n"
            f"  Worst case: {event} in {fmt(self.days_until_stockout_pessimistic)}"
        )


def _daily_consumption(level_series: pd.Series) -> pd.Series:
    """previous_level - current_level, clipped at 0 (resupply days read as 0 consumed)."""
    s = level_series.asfreq("D").ffill()
    consumption = (s.shift(1) - s).clip(lower=0)
    return consumption.dropna()


def _predict_consumable(s, facility, resource, current_level, safety_threshold,
                        forecast_horizon_days, seasonal_periods) -> StockoutPrediction:
    consumption = _daily_consumption(s)

    # Need enough history and non-trivial variation to fit a seasonal model
    if len(consumption) < 60 or consumption.std() == 0:
        avg = max(consumption.mean(), 0.01)
        days_to_stockout = int((current_level - safety_threshold) / avg) if avg > 0 else None
        days_to_stockout = days_to_stockout if days_to_stockout and days_to_stockout >= 0 else None
        return StockoutPrediction(facility, resource, current_level, safety_threshold,
                                   days_to_stockout, days_to_stockout, days_to_stockout,
                                   forecast_horizon_days, kind="consumable")

    fit_result = _fit_ets(consumption, seasonal_periods=seasonal_periods)
    pred = fit_result.get_prediction(start=len(consumption), end=len(consumption) + forecast_horizon_days - 1)
    frame = pred.summary_frame(alpha=0.05)

    mean_consumption = frame["mean"].clip(lower=0).values
    low_consumption = frame["pi_lower"].clip(lower=0).values    # optimistic: less consumption
    high_consumption = frame["pi_upper"].clip(lower=0).values   # pessimistic: more consumption

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
        kind="consumable",
    )


def _predict_capacity(s, facility, resource, current_level, safety_threshold,
                      forecast_horizon_days, seasonal_periods) -> StockoutPrediction:
    """Forecast the available level directly; find when it reaches the safety threshold."""
    if len(s) < 60 or s.std() == 0:
        # too little variation or history to fit; only report if already at/below threshold
        now = 1 if current_level <= safety_threshold else None
        return StockoutPrediction(facility, resource, current_level, safety_threshold,
                                   now, now, now, forecast_horizon_days, kind="capacity")

    fit_result = _fit_ets(s, seasonal_periods=seasonal_periods)
    pred = fit_result.get_prediction(start=len(s), end=len(s) + forecast_horizon_days - 1)
    frame = pred.summary_frame(alpha=0.05)

    mean_path = frame["mean"].values
    best_path = frame["pi_upper"].values    # more capacity free than expected
    worst_path = frame["pi_lower"].values   # less capacity free than expected

    def first_breach(path):
        breach = np.where(path <= safety_threshold)[0]
        return int(breach[0]) + 1 if len(breach) > 0 else None

    return StockoutPrediction(
        facility=facility, resource=resource,
        current_level=current_level, safety_threshold=safety_threshold,
        days_until_stockout_mean=first_breach(mean_path),
        days_until_stockout_optimistic=first_breach(best_path),
        days_until_stockout_pessimistic=first_breach(worst_path),
        forecast_horizon_days=forecast_horizon_days,
        kind="capacity",
    )


def predict_stockout(
    level_series: pd.Series,
    facility: str,
    resource: str,
    safety_threshold: float | None = None,
    forecast_horizon_days: int = 90,
    seasonal_periods: int = 7,
    kind: str | None = None,
) -> StockoutPrediction:
    """
    level_series: date-indexed resource level history for one facility/resource.
    safety_threshold: absolute level considered "critical". Defaults to 10% of
        the observed 90-day max level if not provided.
    kind: "consumable" or "capacity". Inferred from the resource name if omitted.
    """
    kind = kind or resource_kind(resource)
    s = level_series.asfreq("D").ffill()
    current_level = float(s.iloc[-1])

    if safety_threshold is None:
        safety_threshold = 0.10 * s.tail(90).max()

    if kind == "capacity":
        return _predict_capacity(s, facility, resource, current_level, safety_threshold,
                                 forecast_horizon_days, seasonal_periods)
    return _predict_consumable(s, facility, resource, current_level, safety_threshold,
                               forecast_horizon_days, seasonal_periods)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ingestion"))
    from loader import load_resource_csv

    df, _ = load_resource_csv(os.path.join(os.path.dirname(__file__), "../../data/resources_sample.csv"))

    # --- consumable check (should be unchanged from before) ---
    sub = df[(df.facility == "Kisumu Referral Hospital") & (df.resource == "ORS Sachets")]
    full_series = sub.set_index("date")["level"]

    cutoff = pd.Timestamp("2023-04-18")  # a few days into the known suppression window (04-10 to 04-27)
    print(f"Simulating as of {cutoff.date()} (partway through the known suppressed-resupply window)\n")
    print(predict_stockout(full_series.loc[:cutoff], "Kisumu Referral Hospital", "ORS Sachets",
                           forecast_horizon_days=30).summary())

    print("\n--- For comparison, current real-world state (using full history) ---")
    print(predict_stockout(full_series, "Kisumu Referral Hospital", "ORS Sachets",
                           forecast_horizon_days=30).summary())

    # --- capacity check: new beds model vs. the old consumable treatment, 90-day horizon ---
    print("\n--- ICU Beds, 90-day horizon: capacity model vs. old consumable model ---")
    beds = df[(df.facility == "Kisumu Referral Hospital") & (df.resource == "ICU Beds")].set_index("date")["level"]
    for kind in ("capacity", "consumable"):
        r = predict_stockout(beds, "Kisumu Referral Hospital", "ICU Beds",
                             forecast_horizon_days=90, kind=kind)
        print(r.summary() + "\n")