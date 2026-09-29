
"""
Generates synthetic hospital surveillance and resource data for OutbreakWatch
development and testing.

Produces two CSVs:
  - surveillance_sample.csv : daily case counts per facility/condition
  - resources_sample.csv    : daily resource levels (medicine stock, beds) per facility

Includes realistic weekly + annual seasonality, gradual trend, noise, and a
small number of DELIBERATE injected anomalies (outbreak spikes, stock
depletion events) so we can later verify the detection pipeline actually
catches known events. The injected event dates are printed at the end —
keep them, we'll use them as ground truth for validating the system.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(seed=42)

FACILITIES = ["Nairobi County Hospital", "Kisumu Referral Hospital", "Nakuru District Hospital"]
CONDITIONS = ["Malaria", "Cholera", "Respiratory Infections"]

START = date(2022, 1, 1)
END = date(2026, 8, 31)
DAYS = pd.date_range(START, END, freq="D")

# baseline daily case rate per facility/condition (roughly proportional to catchment size)
BASELINE = {
    ("Nairobi County Hospital", "Malaria"): 6.0,
    ("Nairobi County Hospital", "Cholera"): 1.0,
    ("Nairobi County Hospital", "Respiratory Infections"): 9.0,
    ("Kisumu Referral Hospital", "Malaria"): 14.0,   # higher malaria burden near the lake
    ("Kisumu Referral Hospital", "Cholera"): 2.5,
    ("Kisumu Referral Hospital", "Respiratory Infections"): 7.0,
    ("Nakuru District Hospital", "Malaria"): 4.0,
    ("Nakuru District Hospital", "Cholera"): 0.5,
    ("Nakuru District Hospital", "Respiratory Infections"): 6.0,
}

# deliberately injected outbreak windows: (facility, condition, start, length_days, peak_multiplier)
INJECTED_OUTBREAKS = [
    ("Kisumu Referral Hospital", "Cholera", date(2023, 4, 10), 21, 6.0),
    ("Nairobi County Hospital", "Respiratory Infections", date(2024, 7, 1), 28, 4.5),
    ("Nakuru District Hospital", "Malaria", date(2025, 11, 5), 18, 5.0),
    ("Kisumu Referral Hospital", "Malaria", date(2026, 5, 20), 25, 3.5),
]

def seasonal_factor(day_of_year, condition):
    # simple annual seasonality: malaria peaks after long rains (Apr-Jun), cholera
    # peaks in dry-to-wet transition, respiratory peaks in cold season (Jun-Aug)
    if condition == "Malaria":
        peak_day = 150  # ~ late May
    elif condition == "Cholera":
        peak_day = 100  # ~ early April
    else:
        peak_day = 200  # ~ mid July
    return 1.0 + 0.5 * np.cos(2 * np.pi * (day_of_year - peak_day) / 365.25)

def weekly_factor(weekday):
    # slightly lower reporting on weekends (Sat=5, Sun=6)
    return 0.85 if weekday >= 5 else 1.0

def outbreak_multiplier(facility, condition, d):
    for (f, c, start, length, peak_mult) in INJECTED_OUTBREAKS:
        if f == facility and c == condition and start <= d < start + timedelta(days=length):
            t = (d - start).days / length
            # smooth rise-and-fall bump (sine hump) peaking mid-window
            bump = np.sin(np.pi * t) * (peak_mult - 1.0)
            return 1.0 + bump
    return 1.0

rows = []
for facility in FACILITIES:
    for condition in CONDITIONS:
        base = BASELINE[(facility, condition)]
        for d in DAYS:
            doy = d.dayofyear
            seas = seasonal_factor(doy, condition)
            weekly = weekly_factor(d.weekday())
            trend = 1.0 + 0.02 * (d.year - START.year)  # mild year-over-year growth
            outbreak = outbreak_multiplier(facility, condition, d.date())
            expected = base * seas * weekly * trend * outbreak
            noise = rng.poisson(max(expected, 0.1))
            rows.append({
                "date": d.date().isoformat(),
                "facility": facility,
                "condition": condition,
                "case_count": int(noise),
            })

surveillance_df = pd.DataFrame(rows)
surveillance_df.to_csv("surveillance_sample.csv", index=False)

# --- Resource data: medicine stock + bed occupancy ---
RESOURCES = ["ORS Sachets", "Antimalarials", "Respiratory Support Kits", "ICU Beds"]
STARTING_STOCK = {
    "ORS Sachets": 5000,
    "Antimalarials": 3000,
    "Respiratory Support Kits": 400,
    "ICU Beds": 40,   # capacity, not consumable, but tracked the same way
}
DAILY_CONSUMPTION_RATE = {
    "ORS Sachets": 0.04,        # fraction of stock consumed per unit case_count of Cholera
    "Antimalarials": 0.03,      # per unit case_count of Malaria
    "Respiratory Support Kits": 0.02,
    "ICU Beds": 0.01,
}
RESUPPLY_EVERY_DAYS = 14

INJECTED_STOCKOUTS = [
    ("Kisumu Referral Hospital", "ORS Sachets", date(2023, 4, 10), date(2023, 4, 27)),  # coincides with cholera outbreak above
]

res_rows = []
case_lookup = surveillance_df.set_index(["date", "facility", "condition"])["case_count"]

for facility in FACILITIES:
    stock = dict(STARTING_STOCK)
    days_since_resupply = 0
    for d in DAYS:
        days_since_resupply += 1
        ds = d.date().isoformat()

        cholera_cases = case_lookup.get((ds, facility, "Cholera"), 0)
        malaria_cases = case_lookup.get((ds, facility, "Malaria"), 0)
        resp_cases = case_lookup.get((ds, facility, "Respiratory Infections"), 0)

        stock["ORS Sachets"] -= cholera_cases * 10  # ~10 sachets per case
        stock["Antimalarials"] -= malaria_cases * 6
        stock["Respiratory Support Kits"] -= resp_cases * 1
        stock["ICU Beds"] = STARTING_STOCK["ICU Beds"] - int(0.15 * resp_cases)  # occupancy proxy

        # deliberately suppress resupply during the injected Kisumu ORS stockout window
        suppress = any(
            facility == f and res == "ORS Sachets" and d.date() == ev
            for (f, res, ev) in INJECTED_STOCKOUTS
        )

        if days_since_resupply >= RESUPPLY_EVERY_DAYS and not suppress:
            # Resupply tops stock back up TOWARD capacity, it doesn't add on top of
            # whatever's left. (The original version added a fixed amount every cycle
            # regardless of current level, which meant stock grew unbounded over time
            # instead of realistically depleting — caught while building stock-out
            # prediction, since it made every series look permanently well-stocked.)
            for res in ["ORS Sachets", "Antimalarials", "Respiratory Support Kits"]:
                cap = STARTING_STOCK[res]
                stock[res] = min(cap, max(stock[res], 0) + cap * 0.6)
            days_since_resupply = 0

        for res in RESOURCES:
            res_rows.append({
                "date": ds,
                "facility": facility,
                "resource": res,
                "level": max(int(stock[res]), 0),
            })

resources_df = pd.DataFrame(res_rows)
resources_df.to_csv("resources_sample.csv", index=False)

print(f"surveillance_sample.csv: {len(surveillance_df)} rows")
print(f"resources_sample.csv: {len(resources_df)} rows")
print("\nInjected outbreak windows (ground truth for later validation):")
for f, c, start, length, mult in INJECTED_OUTBREAKS:
    print(f"  {f} / {c}: {start} to {start + timedelta(days=length)} (peak x{mult})")
print("\nInjected stock-out event:")
for f, res, ev in INJECTED_STOCKOUTS:
    print(f"  {f} / {res}: suppressed resupply around {ev}")