import pandas as pd
import pytest
from forecaster import forecast, backtest

FACILITY, CONDITION = "Kisumu Referral Hospital", "Malaria"


@pytest.fixture(scope="module")
def series(case_series):
    return case_series(FACILITY, CONDITION)


def test_backtest_rejects_series_that_is_too_short():
    short = pd.Series(range(100), index=pd.date_range("2024-01-01", periods=100, freq="D"))
    with pytest.raises(ValueError, match="too short"):
        backtest(short)


@pytest.mark.slow
def test_forecast_shape_and_ordering(series):
    fc = forecast(series, steps=14)

    assert len(fc) == 14
    assert list(fc.columns) == ["mean", "lower", "upper"]
    assert fc.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert (fc["lower"] <= fc["mean"]).all()
    assert (fc["mean"] <= fc["upper"]).all()
    assert (fc[["mean", "lower", "upper"]] >= 0).all().all()   # case counts can't be negative


@pytest.mark.slow
def test_forecast_is_in_a_sane_range(series):
    fc = forecast(series, steps=14)
    recent = series.iloc[-28:].mean()
    assert abs(fc["mean"].mean() - recent) / recent < 0.5


@pytest.mark.slow
def test_backtest_error_and_interval_coverage(series):
    bt = backtest(series, horizon_days=14, n_folds=6)

    assert bt.n_folds == 6
    assert bt.rmse >= bt.mae
    assert bt.mae < 4.5                 # observed ~3.2 on this series
    assert bt.interval_coverage >= 0.85  # observed ~0.93; target is 0.95