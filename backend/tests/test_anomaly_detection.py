import pandas as pd
import pytest
from detector import detect_all
from ground_truth import OUTBREAKS, QUIET_SERIES

pytestmark = pytest.mark.slow

MAX_DETECTION_DELAY_DAYS = 20     # observed 1 to 13; tighten as the detector improves
MAX_FALSE_ALARM_RATE = 0.005      # observed 0 to 0.18% of days


@pytest.fixture(scope="module")
def flags_for(case_series):
    cache = {}

    def _get(facility, condition):
        key = (facility, condition)
        if key not in cache:
            cache[key] = detect_all(case_series(facility, condition))
        return cache[key]

    return _get


OUTBREAK_IDS = [f"{f.split()[0]}-{c.split()[0]}" for f, c, _, _ in OUTBREAKS]


@pytest.mark.parametrize("facility,condition,start,end", OUTBREAKS, ids=OUTBREAK_IDS)
def test_injected_outbreak_is_detected_early(flags_for, facility, condition, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    alerts = [
        f for f in flags_for(facility, condition)
        if f.severity == "alert" and start <= f.date <= end + pd.Timedelta(days=5)
    ]
    assert alerts, f"no alert-level flag during the {condition} outbreak at {facility}"

    delay = (min(f.date for f in alerts) - start).days
    assert delay <= MAX_DETECTION_DELAY_DAYS


@pytest.mark.parametrize("facility,condition,start,end", OUTBREAKS, ids=OUTBREAK_IDS)
def test_no_alert_storm_outside_the_outbreak(flags_for, case_series, facility, condition, start, end):
    lo = pd.Timestamp(start) - pd.Timedelta(days=10)
    hi = pd.Timestamp(end) + pd.Timedelta(days=10)
    stray_alerts = [
        f for f in flags_for(facility, condition)
        if f.severity == "alert" and not (lo <= f.date <= hi)
    ]
    n_days = len(case_series(facility, condition))
    assert len(stray_alerts) / n_days <= MAX_FALSE_ALARM_RATE


@pytest.mark.parametrize("facility,condition", QUIET_SERIES, ids=[f"{f.split()[0]}-{c}" for f, c in QUIET_SERIES])
def test_series_without_an_outbreak_stays_quiet(flags_for, case_series, facility, condition):
    # these two series were not part of my earlier validation, so the 1% bound is a first guess
    alerts = [f for f in flags_for(facility, condition) if f.severity == "alert"]
    n_days = len(case_series(facility, condition))
    assert len(alerts) / n_days <= 0.01