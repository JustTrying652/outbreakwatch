"""
Ingestion & validation for OutbreakWatch.

Reads raw surveillance/resource CSVs and returns:
  - a clean DataFrame safe for downstream use
  - a ValidationReport describing what was rejected, and what gaps or
    data-quality outliers were flagged (but not silently dropped)

Design principle (from the concept note): validate, don't silently absorb.
Malformed rows are rejected and counted. Missing dates and suspicious jumps
are flagged in the report, not hidden from the caller.
"""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class ValidationReport:
    total_rows_seen: int = 0
    rows_rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)   # reason -> count
    date_gaps: list = field(default_factory=list)           # list of dicts describing gaps
    quality_outliers: list = field(default_factory=list)    # list of dicts describing suspicious jumps

    def summary(self) -> str:
        lines = [
            f"Rows seen: {self.total_rows_seen}",
            f"Rows rejected: {self.rows_rejected}",
        ]
        for reason, count in self.rejection_reasons.items():
            lines.append(f"  - {reason}: {count}")
        lines.append(f"Date gaps flagged: {len(self.date_gaps)}")
        lines.append(f"Quality outliers flagged: {len(self.quality_outliers)}")
        return "\n".join(lines)

    def _reject(self, reason: str, n: int = 1):
        self.rows_rejected += n
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + n


def load_surveillance_csv(path: str) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Expects columns: date, facility, condition, case_count
    Returns (clean_df, report). clean_df is sorted by date and safe to use.
    """
    report = ValidationReport()
    raw = pd.read_csv(path, dtype=str)  # read as str first so we control coercion/validation
    report.total_rows_seen = len(raw)

    required_cols = {"date", "facility", "condition", "case_count"}
    missing_cols = required_cols - set(raw.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df = raw.copy()

    # --- reject rows with missing required fields ---
    missing_mask = df[list(required_cols)].isna().any(axis=1) | (df[list(required_cols)] == "").any(axis=1)
    report._reject("missing required field", int(missing_mask.sum()))
    df = df[~missing_mask]

    # --- validate + coerce date ---
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    bad_date_mask = parsed_dates.isna()
    report._reject("unparseable date", int(bad_date_mask.sum()))
    df = df[~bad_date_mask]
    df["date"] = parsed_dates[~bad_date_mask]

    # --- validate + coerce case_count (must be a non-negative integer) ---
    case_count = pd.to_numeric(df["case_count"], errors="coerce")
    bad_count_mask = case_count.isna() | (case_count < 0) | (case_count != case_count.round())
    report._reject("invalid case_count (non-numeric, negative, or non-integer)", int(bad_count_mask.sum()))
    df = df[~bad_count_mask]
    df["case_count"] = case_count[~bad_count_mask].astype(int)

    df["facility"] = df["facility"].str.strip()
    df["condition"] = df["condition"].str.strip()
    df = df.sort_values(["facility", "condition", "date"]).reset_index(drop=True)

    # --- flag date gaps per facility/condition series ---
    for (facility, condition), group in df.groupby(["facility", "condition"]):
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        present = set(group["date"])
        missing_dates = [d for d in full_range if d not in present]
        if missing_dates:
            report.date_gaps.append({
                "facility": facility,
                "condition": condition,
                "missing_count": len(missing_dates),
                "first_missing": str(missing_dates[0].date()),
                "last_missing": str(missing_dates[-1].date()),
            })

    # --- flag data-quality outliers: single-day jumps far beyond the series' own
    #     historical spread. This is a coarse data-entry sanity check, NOT the
    #     epidemiological anomaly detector (that's a separate, more careful module) ---
    for (facility, condition), group in df.groupby(["facility", "condition"]):
        series = group.set_index("date")["case_count"].asfreq("D")
        mean, std = series.mean(), series.std()
        if std == 0 or pd.isna(std):
            continue
        z = (series - mean) / std
        extreme = series[z.abs() > 6]  # very conservative threshold — data-entry errors, not outbreaks
        for d, val in extreme.items():
            report.quality_outliers.append({
                "facility": facility,
                "condition": condition,
                "date": str(d.date()),
                "case_count": int(val),
                "z_score": round(float(z.loc[d]), 2),
            })

    return df, report


def load_resource_csv(path: str) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Expects columns: date, facility, resource, level
    Returns (clean_df, report).
    """
    report = ValidationReport()
    raw = pd.read_csv(path, dtype=str)
    report.total_rows_seen = len(raw)

    required_cols = {"date", "facility", "resource", "level"}
    missing_cols = required_cols - set(raw.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df = raw.copy()

    missing_mask = df[list(required_cols)].isna().any(axis=1) | (df[list(required_cols)] == "").any(axis=1)
    report._reject("missing required field", int(missing_mask.sum()))
    df = df[~missing_mask]

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    bad_date_mask = parsed_dates.isna()
    report._reject("unparseable date", int(bad_date_mask.sum()))
    df = df[~bad_date_mask]
    df["date"] = parsed_dates[~bad_date_mask]

    level = pd.to_numeric(df["level"], errors="coerce")
    bad_level_mask = level.isna() | (level < 0)
    report._reject("invalid level (non-numeric or negative)", int(bad_level_mask.sum()))
    df = df[~bad_level_mask]
    df["level"] = level[~bad_level_mask].astype(int)

    df["facility"] = df["facility"].str.strip()
    df["resource"] = df["resource"].str.strip()
    df = df.sort_values(["facility", "resource", "date"]).reset_index(drop=True)

    for (facility, resource), group in df.groupby(["facility", "resource"]):
        full_range = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        present = set(group["date"])
        missing_dates = [d for d in full_range if d not in present]
        if missing_dates:
            report.date_gaps.append({
                "facility": facility,
                "resource": resource,
                "missing_count": len(missing_dates),
                "first_missing": str(missing_dates[0].date()),
                "last_missing": str(missing_dates[-1].date()),
            })

    return df, report


if __name__ == "__main__":
    # quick manual smoke test against the sample data generated earlier
    df, report = load_surveillance_csv("../../data/surveillance_sample.csv")
    print("=== Surveillance ===")
    print(report.summary())
    print(df.head())

    res_df, res_report = load_resource_csv("../../data/resources_sample.csv")
    print("\n=== Resources ===")
    print(res_report.summary())
    print(res_df.head())