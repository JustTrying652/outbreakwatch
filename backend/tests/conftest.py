import os
import subprocess
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
APP_DIR = os.path.join(BACKEND_DIR, "app")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

# the app modules import each other by bare name, so make each folder importable
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, BACKEND_DIR)
for sub in ("ingestion", "forecasting", "anomaly_detection", "resource_tracking", "summaries"):
    sys.path.insert(0, os.path.join(APP_DIR, sub))


@pytest.fixture(scope="session", autouse=True)
def sample_data():
    """Paths to the sample CSVs. Regenerates them (seeded, so identical) if missing."""
    surv = os.path.join(DATA_DIR, "surveillance_sample.csv")
    res = os.path.join(DATA_DIR, "resources_sample.csv")
    if not (os.path.exists(surv) and os.path.exists(res)):
        subprocess.run([sys.executable, "generate_sample_data.py"], cwd=DATA_DIR, check=True)
    return surv, res


@pytest.fixture(scope="session")
def surveillance(sample_data):
    from loader import load_surveillance_csv
    df, _ = load_surveillance_csv(sample_data[0])
    return df


@pytest.fixture(scope="session")
def resources(sample_data):
    from loader import load_resource_csv
    df, _ = load_resource_csv(sample_data[1])
    return df


@pytest.fixture(scope="session")
def case_series(surveillance):
    """case_series(facility, condition) -> daily case-count Series."""
    def _get(facility, condition):
        sub = surveillance[(surveillance.facility == facility) & (surveillance.condition == condition)]
        return sub.set_index("date")["case_count"].asfreq("D")
    return _get


@pytest.fixture(scope="session")
def level_series(resources):
    """level_series(facility, resource) -> daily resource-level Series."""
    def _get(facility, resource):
        sub = resources[(resources.facility == facility) & (resources.resource == resource)]
        return sub.set_index("date")["level"].asfreq("D")
    return _get