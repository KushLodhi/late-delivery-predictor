"""Unit tests for timezone normalization, out-of-order scan reconciliation, hub renames, and chronological split."""

from __future__ import annotations

import pandas as pd

from src.config import CANONICAL_HUBS
from src.data_generator import generate_coldchain_dataset
from src.preprocessing import (
    canonicalize_hub_name,
    chronological_split,
    clean_and_engineer_features,
    parse_mixed_timestamp_to_utc,
    reconcile_warehouse_scans,
)
from src.threshold import optimize_threshold, theoretical_bayes_threshold


def test_hub_canonicalization_maps_legacy_codes():
    assert canonicalize_hub_name("DEL-OKH-OLD") == "HUB-DEL-OKH"
    assert canonicalize_hub_name("noi-sec63-legacy") == "HUB-UP-NOI"
    assert canonicalize_hub_name("GGN-HUB-1") == "HUB-HR-GGN"
    assert canonicalize_hub_name("HUB-HR-FBD") == "HUB-HR-FBD"


def test_mixed_timezone_parsing_normalizes_to_same_utc():
    # All four strings represent 2025-08-15 04:00:00 UTC
    s = pd.Series(
        [
            "2025-08-15T09:30:00+05:30",  # explicit IST
            "2025-08-15T04:00:00Z",       # UTC
            "2025-08-15 09:30:00",        # naive IST
            "2025-08-15T08:00:00+04:00",  # misconfigured +04:00 scanner
        ]
    )
    parsed = parse_mixed_timestamp_to_utc(s)
    assert parsed.isna().sum() == 0
    expected = pd.Timestamp("2025-08-15T04:00:00Z")
    assert all(ts == expected for ts in parsed)


def test_reconcile_out_of_order_warehouse_scans():
    scans = pd.DataFrame(
        [
            {"shipment_id": "S1", "hub_raw": "DEL-OKH-OLD", "scan_stage": "ARRIVED", "scan_time_utc": "2025-05-01T06:00:00Z", "recorded_seq": 1},
            {"shipment_id": "S1", "hub_raw": "DEL-OKH-OLD", "scan_stage": "LOADED", "scan_time_utc": "2025-05-01T08:00:00Z", "recorded_seq": 2},
            {"shipment_id": "S1", "hub_raw": "DEL-OKH-OLD", "scan_stage": "TEMP_CHK", "scan_time_utc": "2025-05-01T07:00:00Z", "recorded_seq": 3},
        ]
    )
    summary = reconcile_warehouse_scans(scans)
    assert len(summary) == 1
    assert summary.loc[0, "reconciled_dwell_hours"] == 2.0
    assert summary.loc[0, "reconciled_scan_count"] == 3
    assert summary.loc[0, "scan_order_anomaly"] == 1


def test_chronological_split_has_no_leakage():
    raw_df, scans_df = generate_coldchain_dataset(n_rows=1000, seed=7)
    clean_df = clean_and_engineer_features(raw_df, scans_df)
    assert set(clean_df["origin_hub_id"].unique()).issubset(set(CANONICAL_HUBS.keys()))

    train_df, val_df, test_df = chronological_split(clean_df)
    assert train_df["dispatch_time_utc"].max() <= val_df["dispatch_time_utc"].min()
    assert val_df["dispatch_time_utc"].max() <= test_df["dispatch_time_utc"].min()


def test_cost_matrix_threshold_beats_default():
    import numpy as np

    rng = np.random.default_rng(42)
    y_true = (rng.random(1500) < 0.20).astype(int)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0.15, 0.18, 1500), 0.01, 0.99)
    report = optimize_threshold(y_true, y_prob)
    assert 0.0 < theoretical_bayes_threshold() < 0.20
    assert report["optimal_metrics"]["total_cost_inr"] <= report["default_050_metrics"]["total_cost_inr"]
