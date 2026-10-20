import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.main import app          # loads the data once, at import
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["surveillance_rows"] == 15336
    assert body["resource_rows"] == 20448
    assert body["data_range"][1] == "2026-08-31"


def test_facilities(client):
    body = client.get("/facilities").json()
    assert "Kisumu Referral Hospital" in body["facilities"]
    assert "Malaria" in body["conditions"]
    assert "ICU Beds" in body["resources"]


def test_surveillance_returns_requested_window(client):
    body = client.get("/surveillance/Kisumu Referral Hospital/Malaria", params={"days": 30}).json()
    assert len(body["data"]) == 30
    assert body["data"][-1]["date"] == "2026-08-31"


def test_unknown_names_return_404(client):
    assert client.get("/surveillance/Nowhere Hospital/Malaria").status_code == 404
    assert client.get("/surveillance/Kisumu Referral Hospital/Plague").status_code == 404
    assert client.get("/resources/Kisumu Referral Hospital/Unobtainium").status_code == 404


def test_out_of_range_parameters_are_rejected(client):
    assert client.get("/surveillance/Kisumu Referral Hospital/Malaria", params={"days": 3}).status_code == 422


def test_bad_as_of_date_is_a_400(client):
    r = client.get("/stockout/Kisumu Referral Hospital/ORS Sachets", params={"as_of": "garbage"})
    assert r.status_code == 400


def test_data_quality_report_is_exposed(client):
    body = client.get("/data-quality").json()
    assert body["surveillance"]["rows_rejected"] == 0
    assert body["resources"]["rows_rejected"] == 0


@pytest.mark.slow
def test_anomalies_endpoint_finds_the_cholera_outbreak(client):
    body = client.get("/anomalies/Kisumu Referral Hospital/Cholera").json()
    in_window = [f for f in body["flags"] if "2023-04-10" <= f["date"] <= "2023-05-06" and f["severity"] == "alert"]
    assert in_window


@pytest.mark.slow
def test_stockout_endpoint_reports_beds_as_capacity(client):
    body = client.get("/stockout/Kisumu Referral Hospital/ICU Beds").json()
    assert body["kind"] == "capacity"
    assert body["days_until_stockout"]["expected"] is None