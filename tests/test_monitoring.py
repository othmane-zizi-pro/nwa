import pandas as pd

from src.monitoring import build_input_drift_report, population_stability_index


def test_population_stability_index_zero_for_same_distribution():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    psi = population_stability_index(s, s)
    assert abs(psi) < 1e-9


def test_build_input_drift_report_outputs_expected_columns():
    baseline = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 11, 12, 13, 14]})
    current = pd.DataFrame({"a": [1, 2, 3, 100, 120], "b": [10, 10, 11, 12, 13]})

    report = build_input_drift_report(baseline, current)
    assert set(["Feature", "PSI", "DriftLevel"]).issubset(report.columns)
    assert len(report) == 2
