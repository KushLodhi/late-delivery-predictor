"""Deterministic data cleaning, UTC normalization, scan reconciliation, and feature engineering.

Handles all known DelhiCold Logistics data quality hazards:
  1. Mixed time zones -> normalized to UTC (naive timestamps assumed Asia/Kolkata +05:30).
  2. Out-of-order warehouse scan logs -> sorted chronologically per shipment to recover true dwell time.
  3. Hub renames after the 2025-07-01 network change -> mapped to 12 canonical hub IDs.
  4. Leak-free chronological train / validation / test split.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_HUBS,
    CATEGORICAL_FEATURES,
    HUB_RENAME_CUTOFF_UTC,
    LEGACY_HUB_MAP,
    NUMERIC_FEATURES,
)


def canonicalize_hub_name(hub_code: object) -> str:
    """Map legacy pre-network-change hub codes to canonical 12-hub IDs."""
    if pd.isna(hub_code):
        return "UNKNOWN"
    cleaned = str(hub_code).strip().upper()
    if cleaned in CANONICAL_HUBS:
        return cleaned
    return LEGACY_HUB_MAP.get(cleaned, cleaned)


def parse_mixed_timestamp_to_utc(series: pd.Series) -> pd.Series:
    """Normalize mixed timezone strings (IST +05:30, UTC Z, +04:00, naive IST) to UTC."""
    str_vals = series.astype(str).str.strip()
    # Identify strings that already carry an explicit timezone offset or trailing Z
    has_tz = str_vals.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False)

    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")

    if has_tz.any():
        parsed_tz = pd.to_datetime(str_vals[has_tz], format="mixed", utc=True, errors="coerce")
        result.loc[has_tz] = parsed_tz

    if (~has_tz).any():
        parsed_naive = pd.to_datetime(str_vals[~has_tz], format="mixed", errors="coerce")
        # Naive timestamps originate from local Delhi hub terminals (Asia/Kolkata = UTC+05:30)
        localized = parsed_naive.dt.tz_localize("Asia/Kolkata", ambiguous="NaT", nonexistent="shift_forward")
        result.loc[~has_tz] = localized.dt.tz_convert("UTC")

    return result


def reconcile_warehouse_scans(scans_df: pd.DataFrame) -> pd.DataFrame:
    """Sort out-of-order warehouse scan events chronologically per shipment.

    Computes reconciled dwell hours (last scan minus first scan) and flags shipments
    whose raw recorded sequence did not match chronological scan order.
    """
    if scans_df.empty:
        return pd.DataFrame(columns=["shipment_id", "reconciled_dwell_hours", "reconciled_scan_count", "scan_order_anomaly"])

    df = scans_df.copy()
    df["scan_time_utc"] = parse_mixed_timestamp_to_utc(df["scan_time_utc"])
    df["hub_id"] = df["hub_raw"].map(canonicalize_hub_name)

    # Detect whether incoming scan timestamps were out of chronological order per shipment
    df["_raw_pos"] = np.arange(len(df))
    df_sorted = df.sort_values(["shipment_id", "scan_time_utc", "_raw_pos"])

    grouped = df_sorted.groupby("shipment_id", sort=False)
    first_ts = grouped["scan_time_utc"].first()
    last_ts = grouped["scan_time_utc"].last()
    scan_cnt = grouped["scan_time_utc"].size()

    # If raw_pos is strictly increasing within each shipment after time sort, scans were in order
    is_monotonic = grouped["_raw_pos"].apply(lambda s: bool(s.is_monotonic_increasing))

    summary = pd.DataFrame(
        {
            "shipment_id": first_ts.index,
            "reconciled_dwell_hours": ((last_ts - first_ts).dt.total_seconds() / 3600.0).clip(lower=0.0).round(2).values,
            "reconciled_scan_count": scan_cnt.values,
            "scan_order_anomaly": (~is_monotonic.values).astype(int),
        }
    )
    return summary


def clean_and_engineer_features(
    raw_df: pd.DataFrame,
    scans_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Clean raw shipment records and derive leak-free dispatch-time features.

    Strictly avoids target leakage: `actual_delivery_raw` is never used as a feature.
    """
    df = raw_df.copy()

    # 1. Canonicalize hub names (resolves post-2025-07-01 network rename)
    if "origin_hub_raw" in df.columns:
        df["origin_hub_id"] = df["origin_hub_raw"].map(canonicalize_hub_name)
    elif "origin_hub_id" in df.columns:
        df["origin_hub_id"] = df["origin_hub_id"].map(canonicalize_hub_name)

    if "dest_hub_raw" in df.columns:
        df["dest_hub_id"] = df["dest_hub_raw"].map(canonicalize_hub_name)
    elif "dest_hub_id" in df.columns:
        df["dest_hub_id"] = df["dest_hub_id"].map(canonicalize_hub_name)

    # 2. Normalize timestamps to UTC
    if "dispatch_time_raw" in df.columns:
        df["dispatch_time_utc"] = parse_mixed_timestamp_to_utc(df["dispatch_time_raw"])
    elif "dispatch_time_utc" in df.columns:
        df["dispatch_time_utc"] = parse_mixed_timestamp_to_utc(df["dispatch_time_utc"])

    # 3. Reconcile out-of-order warehouse scan logs if provided
    if scans_df is not None and not scans_df.empty and "shipment_id" in df.columns:
        scan_summary = reconcile_warehouse_scans(scans_df)
        df = df.merge(scan_summary, on="shipment_id", how="left")
        df["warehouse_dwell_hours"] = df["warehouse_dwell_hours"].fillna(df["reconciled_dwell_hours"])
        df["out_of_order_scan_flag"] = np.maximum(
            df["out_of_order_scan_flag"].fillna(0).astype(int),
            df["scan_order_anomaly"].fillna(0).astype(int),
        )
        df.drop(columns=["reconciled_dwell_hours", "reconciled_scan_count", "scan_order_anomaly"], inplace=True)

    # 4. Calendar & network-regime features in local Delhi time (IST = UTC+05:30)
    ist_dt = df["dispatch_time_utc"] + pd.Timedelta(hours=5, minutes=30)
    df["dispatch_hour_ist"] = ist_dt.dt.hour.astype(float)
    df["dispatch_dayofweek"] = ist_dt.dt.dayofweek.astype(float)
    df["is_weekend"] = (ist_dt.dt.dayofweek >= 5).astype(float)
    df["is_peak_hour"] = (
        ((ist_dt.dt.hour >= 8) & (ist_dt.dt.hour <= 11))
        | ((ist_dt.dt.hour >= 17) & (ist_dt.dt.hour <= 21))
    ).astype(float)
    df["is_monsoon_season"] = ist_dt.dt.month.isin([7, 8, 9]).astype(float)
    df["post_network_change"] = (df["dispatch_time_utc"] >= pd.Timestamp(HUB_RENAME_CUTOFF_UTC)).astype(float)

    # 5. Domain cold-chain & operational ratios
    df["temp_deviation_c"] = (df["avg_cabin_temp_c"] - df["temp_setpoint_c"]).abs()
    df["planned_speed_kmph"] = df["distance_km"] / np.maximum(df["planned_transit_hours"], 0.25)
    df["dwell_to_transit_ratio"] = df["warehouse_dwell_hours"] / np.maximum(df["planned_transit_hours"], 0.25)

    # Static hub structural congestion priors
    hub_cong_lookup = {k: float(v["congestion_base"]) for k, v in CANONICAL_HUBS.items()}
    df["origin_hub_congestion_base"] = df["origin_hub_id"].map(hub_cong_lookup).fillna(0.65)
    df["dest_hub_congestion_base"] = df["dest_hub_id"].map(hub_cong_lookup).fillna(0.65)

    if "source_timezone" not in df.columns:
        df["source_timezone"] = "Asia/Kolkata"

    return df


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform strict time-based train / validation / test split to prevent leakage."""
    if not (0.0 < train_frac < 1.0 and 0.0 < val_frac < 1.0 and train_frac + val_frac < 1.0):
        raise ValueError("train_frac + val_frac must be strictly between 0 and 1.")

    ordered = df.sort_values("dispatch_time_utc").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    return train_df, val_df, test_df


def get_feature_columns() -> list[str]:
    """Return exact ordered feature list expected by the scikit-learn pipeline."""
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES
