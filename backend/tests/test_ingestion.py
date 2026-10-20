import pytest
from loader import load_surveillance_csv, load_resource_csv


def test_clean_sample_data_passes_validation(sample_data):
    df, report = load_surveillance_csv(sample_data[0])
    assert report.total_rows_seen == 15336
    assert report.rows_rejected == 0
    assert report.date_gaps == []
    assert len(df) == 15336


def test_clean_resource_data_passes_validation(sample_data):
    df, report = load_resource_csv(sample_data[1])
    assert report.total_rows_seen == 20448
    assert report.rows_rejected == 0
    assert report.date_gaps == []


def test_malformed_rows_are_rejected_and_counted(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "date,facility,condition,case_count\n"
        "2024-01-01,A,Malaria,5\n"      # valid
        "not-a-date,A,Malaria,5\n"      # bad date
        "2024-01-03,A,Malaria,-2\n"     # negative
        "2024-01-04,A,Malaria,2.5\n"    # non-integer
        "2024-01-05,A,Malaria,abc\n"    # non-numeric
        "2024-01-06,,Malaria,4\n"       # missing facility
        "2024-01-07,A,Malaria,7\n"      # valid
    )
    df, report = load_surveillance_csv(str(path))

    assert report.total_rows_seen == 7
    assert report.rows_rejected == 5
    assert len(df) == 2
    assert report.rejection_reasons["missing required field"] == 1
    assert report.rejection_reasons["unparseable date"] == 1
    bad_counts = sum(n for reason, n in report.rejection_reasons.items() if "case_count" in reason)
    assert bad_counts == 3


def test_missing_days_are_flagged_not_hidden(tmp_path):
    path = tmp_path / "gap.csv"
    days = [d for d in range(1, 11) if d != 5]   # day 5 missing
    rows = "".join(f"2024-01-{d:02d},A,Malaria,3\n" for d in days)
    path.write_text("date,facility,condition,case_count\n" + rows)

    df, report = load_surveillance_csv(str(path))
    assert len(df) == 9
    assert len(report.date_gaps) == 1
    gap = report.date_gaps[0]
    assert gap["missing_count"] == 1
    assert gap["first_missing"] == "2024-01-05"


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("date,facility\n2024-01-01,A\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_surveillance_csv(str(path))


def test_negative_resource_level_is_rejected(tmp_path):
    path = tmp_path / "res.csv"
    path.write_text(
        "date,facility,resource,level\n"
        "2024-01-01,A,ORS Sachets,100\n"
        "2024-01-02,A,ORS Sachets,-5\n"
    )
    df, report = load_resource_csv(str(path))
    assert len(df) == 1
    assert report.rows_rejected == 1