"""
Forecasting for OutbreakWatch.

Uses the ETS (Error-Trend-Seasonal) model — the exponential-smoothing family
named in the concept note — to learn each facility/condition series' baseline
seasonality and trend, then project expected load forward WITH prediction
intervals. Every forecast is back-tested on held-out history so we can report
how often the model is actually wrong, rather than presenting a bare number.

Seasonality: fit at weekly period (7). Daily case-count series rarely have
enough clean signal to fit a full annual (365) seasonal cycle reliably with
plain Holt-Winters/ETS — that's better handled with Fourier terms in a SARIMA
or regression model, which is a reasonable future extension. For now, trend
absorbs slower-moving (e.g. seasonal-of-year) movement, and weekly seasonality
captures the day-of-week reporting pattern. This tradeoff is intentional and
documented here rather than silently assumed.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


@dataclass
class BacktestResult:
    n_folds: int
    horizon_days: int
    mae: float
    rmse: float
    mape: float | None       # None if series has zero-heavy history (MAPE undefined/unstable)
    interval_coverage: float  # fraction of actuals that fell within the predicted interval

    def summary(self) -> str:
        mape_str = f"{self.mape:.1f}%" if self.mape is not None else "n/a (too many zero-actuals)"
        return (
            f"Backtest over {self.n_folds} folds, {self.horizon_days}-day horizon:\n"
            f"  MAE:  {self.mae:.2f} cases/day\n"
            f"  RMSE: {self.rmse:.2f} cases/day\n"
            f"  MAPE: {mape_str}\n"
            f"  95% interval coverage: {self.interval_coverage * 100:.1f}% "
            f"(target ~95%)"
        )


def _fit_ets(train_series: pd.Series, seasonal_periods: int = 7) -> ETSModel:
    """Fit an additive-error, damped-trend, additive-seasonal ETS model."""
    model = ETSModel(
        train_series,
        error="add",
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=seasonal_periods,
    )
    return model.fit(disp=False)


def forecast(series: pd.Series, steps: int = 14, alpha: float = 0.05, seasonal_periods: int = 7) -> pd.DataFrame:
    """
    Fit on the full series and forecast `steps` days ahead.
    Returns a DataFrame indexed by date with columns: mean, lower, upper.
    Negative values are clipped to 0 (case counts can't be negative).
    """
    fit_result = _fit_ets(series, seasonal_periods=seasonal_periods)
    pred = fit_result.get_prediction(
        start=len(series), end=len(series) + steps - 1
    )
    frame = pred.summary_frame(alpha=alpha)

    future_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
    result = pd.DataFrame({
        "mean": frame["mean"].values,
        "lower": frame["pi_lower"].values,
        "upper": frame["pi_upper"].values,
    }, index=future_index)

    result[["mean", "lower", "upper"]] = result[["mean", "lower", "upper"]].clip(lower=0)
    return result


def backtest(series: pd.Series, horizon_days: int = 14, n_folds: int = 6,
             min_train_days: int = 365, seasonal_periods: int = 7) -> BacktestResult:
    """
    Rolling-origin back-test: repeatedly fit on a growing training window,
    forecast `horizon_days` ahead, and compare to the actual held-out values.
    Reports MAE, RMSE, MAPE, and what fraction of actuals fell within the
    predicted 95% interval (should land close to 95% if intervals are honest).
    """
    n = len(series)
    if n < min_train_days + horizon_days * n_folds:
        raise ValueError(
            f"Series too short for {n_folds} folds of {horizon_days}-day backtesting "
            f"with a {min_train_days}-day minimum training window."
        )

    fold_starts = np.linspace(min_train_days, n - horizon_days, n_folds, dtype=int)

    all_errors, all_sq_errors, all_pct_errors, coverage_hits = [], [], [], []

    for train_end in fold_starts:
        train = series.iloc[:train_end]
        actual = series.iloc[train_end: train_end + horizon_days]
        if len(actual) < horizon_days:
            continue

        fit_result = _fit_ets(train, seasonal_periods=seasonal_periods)
        pred = fit_result.get_prediction(start=train_end, end=train_end + horizon_days - 1)
        frame = pred.summary_frame(alpha=0.05)

        forecast_mean = frame["mean"].values.clip(min=0)
        lower = frame["pi_lower"].values.clip(min=0)
        upper = frame["pi_upper"].values.clip(min=0)
        actual_vals = actual.values

        errors = actual_vals - forecast_mean
        all_errors.extend(np.abs(errors))
        all_sq_errors.extend(errors ** 2)

        nonzero_mask = actual_vals > 0
        if nonzero_mask.any():
            pct_err = np.abs(errors[nonzero_mask]) / actual_vals[nonzero_mask]
            all_pct_errors.extend(pct_err)

        hits = (actual_vals >= lower) & (actual_vals <= upper)
        coverage_hits.extend(hits)

    mae = float(np.mean(all_errors))
    rmse = float(np.sqrt(np.mean(all_sq_errors)))
    mape = float(np.mean(all_pct_errors) * 100) if len(all_pct_errors) >= len(all_errors) * 0.5 else None
    coverage = float(np.mean(coverage_hits))

    return BacktestResult(
        n_folds=n_folds, horizon_days=horizon_days,
        mae=mae, rmse=rmse, mape=mape, interval_coverage=coverage,
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../ingestion")
    from loader import load_surveillance_csv

    df, _ = load_surveillance_csv("../../data/surveillance_sample.csv")

    # pick one series to demo: Kisumu Malaria (has a 2026 outbreak near the end of history)
    sub = df[(df.facility == "Kisumu Referral Hospital") & (df.condition == "Malaria")]
    series = sub.set_index("date")["case_count"].asfreq("D").fillna(0)

    print(f"Series length: {len(series)} days ({series.index[0].date()} to {series.index[-1].date()})\n")

    print("=== Back-test (does the model's own track record hold up?) ===")
    bt = backtest(series, horizon_days=14, n_folds=6)
    print(bt.summary())

    print("\n=== 14-day forward forecast from end of series ===")
    fc = forecast(series, steps=14)
    print(fc.round(1))