"""Unit tests for PSI and KS-test feature drift monitoring."""

from __future__ import annotations

import logging

import numpy as np

from src.data_generator import generate_coldchain_dataset
from src.drift import compute_reference_stats, detect_drift
from src.preprocessing import clean_and_engineer_features


def test_drift_monitor_flags_shifted_features_and_logs_warning(caplog):
    raw_df, scans_df = generate_coldchain_dataset(n_rows=3000, seed=99)
    clean_df = clean_and_engineer_features(raw_df, scans_df)

    ref_stats = compute_reference_stats(clean_df.iloc[:2000])

    # Unshifted sample drawn from same distribution should remain healthy
    in_dist = clean_df.iloc[:2000].sample(n=400, random_state=1)
    healthy_report = detect_drift(in_dist, ref_stats, log_warnings=False)
    assert healthy_report["overall_status"] in ("HEALTHY", "MONITOR")
    assert healthy_report["drifted_features"] == []

    # Heavily shifted dwell time & congestion should trigger DRIFT_WARNING + log warning
    shifted = clean_df.iloc[2000:].copy()
    shifted["warehouse_dwell_hours"] = shifted["warehouse_dwell_hours"].fillna(1.2) + 3.5
    shifted["route_congestion_index"] = np.clip(shifted["route_congestion_index"].fillna(0.6) + 0.30, 0, 1)

    with caplog.at_level(logging.WARNING, logger="delhicold.drift"):
        drift_report = detect_drift(shifted, ref_stats, log_warnings=True)

    assert drift_report["overall_status"] == "DRIFT_WARNING"
    assert "warehouse_dwell_hours" in drift_report["drifted_features"]
    assert "route_congestion_index" in drift_report["drifted_features"]
    assert any("DATA DRIFT WARNING" in rec.message for rec in caplog.records)
