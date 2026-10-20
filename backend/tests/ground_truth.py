"""Known events injected by data/generate_sample_data.py. The tests check the system finds them."""

# (facility, condition, start, end)
OUTBREAKS = [
    ("Kisumu Referral Hospital", "Cholera", "2023-04-10", "2023-05-01"),
    ("Nairobi County Hospital", "Respiratory Infections", "2024-07-01", "2024-07-29"),
    ("Nakuru District Hospital", "Malaria", "2025-11-05", "2025-11-23"),
    ("Kisumu Referral Hospital", "Malaria", "2026-05-20", "2026-06-14"),
]

# series with NO injected outbreak: they should stay quiet
QUIET_SERIES = [
    ("Nairobi County Hospital", "Malaria"),
    ("Nakuru District Hospital", "Cholera"),
]

# resupply is suppressed here, so ORS stock falls fast during the cholera outbreak
STOCKOUT_FACILITY = "Kisumu Referral Hospital"
STOCKOUT_RESOURCE = "ORS Sachets"
REPLAY_DATE = "2023-04-18"   # partway through the suppression window