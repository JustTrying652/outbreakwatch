"""
OutbreakWatch API.

Thin HTTP layer over the five validated modules (ingestion, forecasting,
anomaly detection, resource tracking, summaries). No statistics live here:
this file loads data once, calls the modules, caches the slow ETS fits, and
returns JSON.

Run from the backend/ folder (venv active):
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive API explorer.
"""

import os
import sys
import warnings
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", message="No frequency information")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "..", "data")

# The modules import each other with bare names, so make each folder importable
for sub in ("ingestion", "forecasting", "anomaly_detection", "resource_tracking", "summaries"):
    sys.path.insert(0, os.path.join(APP_DIR, sub))

from loader import load_surveillance_csv, load_resource_csv
from forecaster import forecast, backtest
from detector import detect_all
from tracker import predict_stockout
from generator import summarize_condition, summarize_resource, facility_situation_report


# ---------- load data once at startup ----------

SURV, SURV_REPORT = load_surveillance_csv(os.path.join(DATA_DIR, "surveillance_sample.csv"))
RES, RES_REPORT = load_resource_csv(os.path.join(DATA_DIR, "resources_sample.csv"))

FACILITIES = sorted(SURV["facility"].unique())
CONDITIONS = sorted(SURV["condition"].unique())
RESOURCES = sorted(RES["resource"].unique())


# ---------- app ----------

app = FastAPI(title="OutbreakWatch API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- validation + cached computation ----------

def _check_condition(facility: str, condition: str):
    if facility not in FACILITIES:
        raise HTTPException(404, f"Unknown facility: {facility}. Options: {FACILITIES}")
    if condition not in CONDITIONS:
        raise HTTPException(404, f"Unknown condition: {condition}. Options: {CONDITIONS}")


def _check_resource(facility: str, resource: str):
    if facility not in FACILITIES:
        raise HTTPException(404, f"Unknown facility: {facility}. Options: {FACILITIES}")
    if resource not in RESOURCES:
        raise HTTPException(404, f"Unknown resource: {resource}. Options: {RESOURCES}")


@lru_cache(maxsize=None)
def _case_series(facility: str, condition: str) -> pd.Series:
    sub = SURV[(SURV.facility == facility) & (SURV.condition == condition)]
    return sub.set_index("date")["case_count"].asfreq("D").fillna(0)


@lru_cache(maxsize=None)
def _level_series(facility: str, resource: str) -> pd.Series:
    sub = RES[(RES.facility == facility) & (RES.resource == resource)]
    return sub.set_index("date")["level"].asfreq("D").ffill()


# ETS fits are the slow part (seconds each), so results are cached per series.
# Callers must not mutate the cached objects.

@lru_cache(maxsize=None)
def _cached_forecast(facility: str, condition: str, steps: int) -> pd.DataFrame:
    return forecast(_case_series(facility, condition), steps=steps)


@lru_cache(maxsize=None)
def _cached_flags(facility: str, condition: str) -> list:
    return detect_all(_case_series(facility, condition))


@lru_cache(maxsize=None)
def _cached_backtest(facility: str, condition: str):
    return backtest(_case_series(facility, condition), horizon_days=14, n_folds=6)


@lru_cache(maxsize=None)
def _cached_stockout(facility: str, resource: str, horizon: int, as_of: str | None):
    series = _level_series(facility, resource)
    if as_of:
        series = series.loc[: pd.Timestamp(as_of)]
    return predict_stockout(series, facility, resource, forecast_horizon_days=horizon)


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "surveillance_rows": len(SURV),
        "resource_rows": len(RES),
        "data_range": [str(SURV["date"].min().date()), str(SURV["date"].max().date())],
    }


@app.get("/facilities")
def list_facilities():
    return {"facilities": FACILITIES, "conditions": CONDITIONS, "resources": RESOURCES}


@app.get("/data-quality")
def data_quality():
    """Surfaces the ingestion validation reports, so nothing is silently absorbed."""
    return {
        "surveillance": {
            "rows_seen": SURV_REPORT.total_rows_seen,
            "rows_rejected": SURV_REPORT.rows_rejected,
            "rejection_reasons": SURV_REPORT.rejection_reasons,
            "date_gaps": SURV_REPORT.date_gaps,
            "quality_outliers": SURV_REPORT.quality_outliers,
        },
        "resources": {
            "rows_seen": RES_REPORT.total_rows_seen,
            "rows_rejected": RES_REPORT.rows_rejected,
            "rejection_reasons": RES_REPORT.rejection_reasons,
            "date_gaps": RES_REPORT.date_gaps,
        },
    }


@app.get("/surveillance/{facility}/{condition}")
def get_surveillance(facility: str, condition: str, days: int = Query(90, ge=7, le=2000)):
    _check_condition(facility, condition)
    series = _case_series(facility, condition).iloc[-days:]
    return {
        "facility": facility,
        "condition": condition,
        "data": [{"date": d.date().isoformat(), "case_count": int(v)} for d, v in series.items()],
    }


@app.get("/forecast/{facility}/{condition}")
def get_forecast(facility: str, condition: str, steps: int = Query(14, ge=1, le=60)):
    _check_condition(facility, condition)
    fc = _cached_forecast(facility, condition, steps)
    return {
        "facility": facility,
        "condition": condition,
        "interval": "95%",
        "forecast": [
            {
                "date": d.date().isoformat(),
                "mean": round(float(r["mean"]), 2),
                "lower": round(float(r["lower"]), 2),
                "upper": round(float(r["upper"]), 2),
            }
            for d, r in fc.iterrows()
        ],
    }


@app.get("/backtest/{facility}/{condition}")
def get_backtest(facility: str, condition: str):
    _check_condition(facility, condition)
    bt = _cached_backtest(facility, condition)
    return {
        "facility": facility,
        "condition": condition,
        "n_folds": bt.n_folds,
        "horizon_days": bt.horizon_days,
        "mae": round(bt.mae, 2),
        "rmse": round(bt.rmse, 2),
        "mape_percent": round(bt.mape, 1) if bt.mape is not None else None,
        "interval_coverage": round(bt.interval_coverage, 3),
    }


@app.get("/anomalies/{facility}/{condition}")
def get_anomalies(
    facility: str,
    condition: str,
    since_days: int | None = Query(None, ge=1, le=2000, description="Only flags from the last N days"),
):
    _check_condition(facility, condition)
    flags = _cached_flags(facility, condition)
    if since_days is not None:
        cutoff = _case_series(facility, condition).index[-1] - pd.Timedelta(days=since_days)
        flags = [f for f in flags if f.date >= cutoff]
    return {
        "facility": facility,
        "condition": condition,
        "count": len(flags),
        "flags": [
            {
                "date": f.date.date().isoformat(),
                "method": f.method,
                "severity": f.severity,
                "value": float(f.value),
                "expected": round(float(f.expected), 2),
                "score": round(float(f.score), 2),
            }
            for f in flags
        ],
    }


@app.get("/resources/{facility}/{resource}")
def get_resource_history(facility: str, resource: str, days: int = Query(90, ge=7, le=2000)):
    _check_resource(facility, resource)
    series = _level_series(facility, resource).iloc[-days:]
    return {
        "facility": facility,
        "resource": resource,
        "data": [{"date": d.date().isoformat(), "level": int(v)} for d, v in series.items()],
    }


@app.get("/stockout/{facility}/{resource}")
def get_stockout(
    facility: str,
    resource: str,
    horizon: int = Query(30, ge=7, le=180),
    as_of: str | None = Query(None, description="YYYY-MM-DD. Predict as if this were today (for replaying past events)"),
):
    _check_resource(facility, resource)
    if as_of:
        try:
            pd.Timestamp(as_of)
        except ValueError:
            raise HTTPException(400, "as_of must be a date like 2023-04-18")
    p = _cached_stockout(facility, resource, horizon, as_of)
    return {
        "facility": p.facility,
        "resource": p.resource,
        "as_of": as_of,
        "Kind": p.kind,
        "current_level": p.current_level,
        "safety_threshold": round(p.safety_threshold, 1),
        "horizon_days": p.forecast_horizon_days,
        "days_until_stockout": {
            "expected": p.days_until_stockout_mean,
            "best_case": p.days_until_stockout_optimistic,
            "worst_case": p.days_until_stockout_pessimistic,
        },
        "summary": summarize_resource(p),
    }


@app.get("/summary/{facility}")
def get_summary(facility: str):
    """Full plain-language situation report. First call can take a minute (several ETS fits); later calls are cached."""
    if facility not in FACILITIES:
        raise HTTPException(404, f"Unknown facility: {facility}. Options: {FACILITIES}")

    condition_summaries = []
    for condition in CONDITIONS:
        series = _case_series(facility, condition)
        fc = _cached_forecast(facility, condition, 14)
        cutoff = series.index[-1] - pd.Timedelta(days=14)
        recent_flags = [f for f in _cached_flags(facility, condition) if f.date >= cutoff]
        condition_summaries.append(summarize_condition(facility, condition, series, fc, recent_flags))

    resource_summaries = [
        summarize_resource(_cached_stockout(facility, r, 30, None)) for r in RESOURCES
    ]

    return {
        "facility": facility,
        "report": facility_situation_report(facility, condition_summaries, resource_summaries),
        "condition_summaries": condition_summaries,
        "resource_summaries": resource_summaries,
    }