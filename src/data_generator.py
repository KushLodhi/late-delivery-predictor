"""Synthetic data generator for DelhiCold Logistics (12 Delhi/NCR hubs).

Generates realistic shipment, route, vehicle, and warehouse scan records
with intentional real-world data quality issues:
  1. Mixed timestamp time zones (Asia/Kolkata +05:30, UTC Z, naive IST, +04:00 misconfigured scanner).
  2. Out-of-order warehouse scan events (arriving out of chronological sequence).
  3. Hub renames after the 2025-07-01 network restructuring cutoff.
  4. Missing values in sensor / vehicle / dwell fields (~2-4%).
  5. Realistic class imbalance (~18% late deliveries).

Supports scaling up to 1,500,000+ rows via --rows CLI argument (default: 100,000 for fast setup).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    CANONICAL_HUBS,
    CARGO_TYPES,
    DATA_DIR,
    HUB_RENAME_CUTOFF_UTC,
    LEGACY_HUB_MAP,
    RANDOM_STATE,
    VEHICLE_TYPES,
    WEATHER_CONDITIONS,
)

# Inverse mapping: canonical -> list of legacy codes
_CANONICAL_TO_LEGACY = {}
for legacy_code, canonical_code in LEGACY_HUB_MAP.items():
    _CANONICAL_TO_LEGACY.setdefault(canonical_code, []).append(legacy_code)

_CARGO_TEMP_SETPOINT = {
    "vaccines_pharma": 4.0,
    "fresh_dairy": 3.0,
    "frozen_meat": -18.0,
    "fresh_produce": 7.0,
    "ice_cream_confectionery": -20.0,
}

_CARGO_LATE_BIAS = {
    "vaccines_pharma": -0.25,  # priority handling
    "fresh_dairy": 0.15,
    "frozen_meat": 0.10,
    "fresh_produce": 0.30,
    "ice_cream_confectionery": 0.20,
}

_WEATHER_LATE_BIAS = {
    "clear": -0.35,
    "haze_smog": 0.25,
    "heavy_rain_monsoon": 1.05,
    "extreme_heatwave": 0.65,
    "dense_fog": 1.15,
}

_VEHICLE_LATE_BIAS = {
    "reefer_van_1t": -0.15,
    "reefer_truck_3t": 0.05,
    "reefer_truck_7t": 0.35,
    "insulated_lcv": 0.20,
}


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a)) * 1.32  # road circuity factor for Delhi NCR


def _format_mixed_timezones(utc_series: pd.Series, tz_choices: np.ndarray) -> pd.Series:
    """Format UTC timestamps into mixed timezone strings to simulate heterogeneous hub scanners."""
    utc_dt = pd.to_datetime(utc_series, utc=True)
    out = np.empty(len(utc_dt), dtype=object)

    ist_mask = tz_choices == "Asia/Kolkata"
    utc_mask = tz_choices == "UTC"
    naive_mask = tz_choices == "NAIVE_IST"
    gulf_mask = tz_choices == "Asia/Dubai"

    if ist_mask.any():
        ist_vals = (utc_dt[ist_mask] + pd.Timedelta(hours=5, minutes=30)).dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")
        out[ist_mask] = ist_vals.values
    if utc_mask.any():
        utc_vals = utc_dt[utc_mask].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out[utc_mask] = utc_vals.values
    if naive_mask.any():
        naive_vals = (utc_dt[naive_mask] + pd.Timedelta(hours=5, minutes=30)).dt.strftime("%Y-%m-%d %H:%M:%S")
        out[naive_mask] = naive_vals.values
    if gulf_mask.any():
        gulf_vals = (utc_dt[gulf_mask] + pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%dT%H:%M:%S+04:00")
        out[gulf_mask] = gulf_vals.values

    return pd.Series(out, index=utc_series.index)


def generate_coldchain_dataset(
    n_rows: int = 100_000,
    seed: int = RANDOM_STATE,
    include_scan_events: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic Delhi cold-chain shipment records and warehouse scan logs.

    Returns:
        shipments_df: Main shipment/route/vehicle table with raw dirty fields.
        scans_df: Warehouse scan event table (with intentional out-of-order scans).
    """
    rng = np.random.default_rng(seed)
    hub_ids = list(CANONICAL_HUBS.keys())
    n_hubs = len(hub_ids)

    # Chronological timeline spanning 12 months: 2025-01-01 to 2025-12-31
    start_ts = pd.Timestamp("2025-01-01T00:00:00Z").value // 10**9
    end_ts = pd.Timestamp("2025-12-31T23:59:59Z").value // 10**9
    dispatch_epoch = rng.integers(start_ts, end_ts, size=n_rows)
    dispatch_epoch.sort()
    dispatch_utc = pd.to_datetime(dispatch_epoch, unit="s", utc=True)

    # Origin & destination hubs (distinct)
    origin_idx = rng.integers(0, n_hubs, size=n_rows)
    dest_idx = (origin_idx + rng.integers(1, n_hubs, size=n_rows)) % n_hubs
    origin_canonical = np.array([hub_ids[i] for i in origin_idx])
    dest_canonical = np.array([hub_ids[i] for i in dest_idx])

    lats = np.array([CANONICAL_HUBS[h]["lat"] for h in hub_ids])
    lons = np.array([CANONICAL_HUBS[h]["lon"] for h in hub_ids])
    cong_base = np.array([CANONICAL_HUBS[h]["congestion_base"] for h in hub_ids])

    base_dist = _haversine_km(lats[origin_idx], lons[origin_idx], lats[dest_idx], lons[dest_idx])
    # Add realistic urban routing variance & occasional longer NCR regional leg
    distance_km = np.clip(base_dist + rng.normal(8.0, 6.0, size=n_rows), 6.0, 140.0).round(2)

    cargo_type = rng.choice(CARGO_TYPES, size=n_rows, p=[0.18, 0.28, 0.16, 0.24, 0.14])
    vehicle_type = rng.choice(VEHICLE_TYPES, size=n_rows, p=[0.30, 0.35, 0.20, 0.15])

    # Seasonal weather distribution based on Delhi calendar month
    months = dispatch_utc.month.values
    weather = np.empty(n_rows, dtype=object)
    for m in range(1, 13):
        mask = months == m
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        if m in (12, 1):  # Winter fog + smog
            probs = [0.25, 0.30, 0.03, 0.02, 0.40]
        elif m in (4, 5, 6):  # Extreme summer heatwave
            probs = [0.35, 0.20, 0.05, 0.38, 0.02]
        elif m in (7, 8, 9):  # Monsoon
            probs = [0.30, 0.15, 0.45, 0.08, 0.02]
        elif m in (10, 11):  # Post-monsoon smog
            probs = [0.32, 0.50, 0.05, 0.05, 0.08]
        else:  # Feb-Mar pleasant
            probs = [0.65, 0.22, 0.05, 0.05, 0.03]
        weather[mask] = rng.choice(WEATHER_CONDITIONS, size=cnt, p=probs)

    # IST hour of dispatch
    ist_dt = dispatch_utc + pd.Timedelta(hours=5, minutes=30)
    ist_hour = ist_dt.hour.values
    ist_dow = ist_dt.dayofweek.values
    is_peak = ((ist_hour >= 8) & (ist_hour <= 11)) | ((ist_hour >= 17) & (ist_hour <= 21))

    # Route congestion index (0.1 to 1.0)
    origin_cong = cong_base[origin_idx]
    dest_cong = cong_base[dest_idx]
    route_congestion = np.clip(
        0.5 * (origin_cong + dest_cong)
        + 0.14 * is_peak.astype(float)
        - 0.06 * (ist_dow >= 5).astype(float)
        + rng.normal(0.0, 0.08, size=n_rows),
        0.10,
        1.00,
    ).round(3)

    # Vehicle & cargo physical features
    vehicle_age_years = np.clip(rng.gamma(shape=2.4, scale=1.8, size=n_rows), 0.2, 14.0).round(1)
    payload_weight_kg = np.clip(rng.normal(1800, 950, size=n_rows), 150, 6800).round(1)

    temp_setpoint_c = np.array([_CARGO_TEMP_SETPOINT[c] for c in cargo_type])
    # Older vehicles & heatwaves increase cabin temperature excursion
    temp_drift = (
        rng.normal(0.3, 1.1, size=n_rows)
        + 0.15 * (vehicle_age_years > 6.0)
        + 0.85 * (weather == "extreme_heatwave")
    )
    avg_cabin_temp_c = (temp_setpoint_c + np.abs(temp_drift)).round(2)
    temp_deviation_c = np.abs(avg_cabin_temp_c - temp_setpoint_c).round(2)

    # Warehouse dwell time & scan events
    warehouse_dwell_hours = np.clip(
        rng.gamma(shape=2.0, scale=0.65, size=n_rows)
        + 0.4 * (origin_cong > 0.75)
        + 0.25 * is_peak.astype(float),
        0.15,
        8.5,
    ).round(2)

    # Planned transit hours (SLA promised to customer)
    planned_speed = rng.uniform(22.0, 34.0, size=n_rows)
    planned_transit_hours = np.clip(distance_km / planned_speed + 0.65, 1.0, 10.0).round(2)

    # Scan count (typically 3-6 warehouse/gate scans) and out-of-order scan anomaly (~12%)
    scan_count = rng.integers(3, 8, size=n_rows)
    out_of_order_scan_flag = (rng.random(n_rows) < 0.12).astype(int)

    # Ground-truth late delivery probability via realistic non-linear log-odds
    cargo_bias = np.array([_CARGO_LATE_BIAS[c] for c in cargo_type])
    weather_bias = np.array([_WEATHER_LATE_BIAS[w] for w in weather])
    vehicle_bias = np.array([_VEHICLE_LATE_BIAS[v] for v in vehicle_type])
    dwell_ratio = warehouse_dwell_hours / np.maximum(planned_transit_hours, 0.5)

    logits = (
        -4.35
        + 1.95 * (route_congestion - 0.55)
        + 0.92 * np.maximum(warehouse_dwell_hours - 1.2, 0.0)
        + 0.80 * dwell_ratio
        + 0.34 * np.maximum(temp_deviation_c - 1.5, 0.0)
        + 0.11 * np.maximum(vehicle_age_years - 4.0, 0.0)
        + 0.015 * np.maximum(distance_km - 35.0, 0.0)
        + 0.55 * is_peak.astype(float)
        + 0.45 * out_of_order_scan_flag
        + cargo_bias
        + weather_bias
        + vehicle_bias
        + 0.65 * ((weather == "heavy_rain_monsoon") & (route_congestion > 0.75)).astype(float)
        + rng.normal(0.0, 0.45, size=n_rows)
    )
    late_prob = 1.0 / (1.0 + np.exp(-logits))
    is_late = (rng.random(n_rows) < late_prob).astype(int)

    # Actual delivery timestamp (UTC)
    delay_hours = np.where(
        is_late == 1,
        rng.uniform(0.35, 3.8, size=n_rows),
        -rng.uniform(0.05, 0.65, size=n_rows),
    )
    planned_delivery_utc = dispatch_utc + pd.to_timedelta(planned_transit_hours, unit="h")
    actual_delivery_utc = planned_delivery_utc + pd.to_timedelta(delay_hours, unit="h")

    # Inject Problem 1: Hub renames prior to 2025-07-01 network change
    cutoff = pd.Timestamp(HUB_RENAME_CUTOFF_UTC)
    pre_change = dispatch_utc < cutoff
    origin_raw = origin_canonical.astype(object)
    dest_raw = dest_canonical.astype(object)
    rename_mask_o = pre_change & (rng.random(n_rows) < 0.75)
    rename_mask_d = pre_change & (rng.random(n_rows) < 0.75)

    for idx in np.where(rename_mask_o)[0]:
        origin_raw[idx] = _CANONICAL_TO_LEGACY[origin_canonical[idx]][0]
    for idx in np.where(rename_mask_d)[0]:
        dest_raw[idx] = _CANONICAL_TO_LEGACY[dest_canonical[idx]][0]

    # Inject Problem 2: Mixed timezones across hubs
    tz_choices = rng.choice(
        ["Asia/Kolkata", "UTC", "NAIVE_IST", "Asia/Dubai"],
        size=n_rows,
        p=[0.50, 0.25, 0.20, 0.05],
    )
    dispatch_raw_str = _format_mixed_timezones(pd.Series(dispatch_utc), tz_choices)
    planned_raw_str = _format_mixed_timezones(pd.Series(planned_delivery_utc), tz_choices)
    actual_raw_str = _format_mixed_timezones(pd.Series(actual_delivery_utc), tz_choices)

    shipment_ids = [f"SHP-{i + 1:07d}" for i in range(n_rows)]
    vehicle_ids = [f"DL1C-{rng.integers(1000, 9999)}" for _ in range(n_rows)]

    # Inject Problem 3: Realistic missingness (~2.5%) in sensor/operational columns
    def _inject_nans(arr: np.ndarray, frac: float = 0.025) -> np.ndarray:
        out = arr.astype(float).copy()
        mask = rng.random(len(out)) < frac
        out[mask] = np.nan
        return out

    shipments_df = pd.DataFrame(
        {
            "shipment_id": shipment_ids,
            "vehicle_id": vehicle_ids,
            "origin_hub_raw": origin_raw,
            "dest_hub_raw": dest_raw,
            "dispatch_time_raw": dispatch_raw_str.values,
            "planned_delivery_raw": planned_raw_str.values,
            "actual_delivery_raw": actual_raw_str.values,
            "source_timezone": tz_choices,
            "cargo_type": cargo_type,
            "vehicle_type": vehicle_type,
            "weather_condition": weather,
            "distance_km": distance_km,
            "planned_transit_hours": planned_transit_hours,
            "warehouse_dwell_hours": _inject_nans(warehouse_dwell_hours, 0.03),
            "scan_count": scan_count,
            "out_of_order_scan_flag": out_of_order_scan_flag,
            "temp_setpoint_c": temp_setpoint_c,
            "avg_cabin_temp_c": _inject_nans(avg_cabin_temp_c, 0.035),
            "vehicle_age_years": _inject_nans(vehicle_age_years, 0.02),
            "payload_weight_kg": _inject_nans(payload_weight_kg, 0.02),
            "route_congestion_index": _inject_nans(route_congestion, 0.025),
            "is_late": is_late,
        }
    )

    # Optional granular warehouse scan log (sample of shipments to keep disk footprint manageable)
    if include_scan_events:
        sample_n = min(n_rows, 5_000)
        scan_rows = []
        stages = ["ARRIVED_ORIGIN_HUB", "COLD_STORAGE_IN", "QUALITY_TEMP_CHECK", "LOADED_REEFER", "DISPATCHED_GATE"]
        for i in range(sample_n):
            base_t = dispatch_utc[i] - pd.Timedelta(hours=float(warehouse_dwell_hours[i]))
            offsets = np.sort(rng.uniform(0.0, float(warehouse_dwell_hours[i]), size=len(stages)))
            events = [
                {
                    "shipment_id": shipment_ids[i],
                    "hub_raw": origin_raw[i],
                    "scan_stage": stages[s],
                    "scan_time_utc": (base_t + pd.Timedelta(hours=float(offsets[s]))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "recorded_seq": s + 1,
                }
                for s in range(len(stages))
            ]
            # Deliberately scramble scan order for out-of-order shipments
            if out_of_order_scan_flag[i] == 1:
                events[1], events[3] = events[3], events[1]
            scan_rows.extend(events)
        scans_df = pd.DataFrame(scan_rows)
    else:
        scans_df = pd.DataFrame(columns=["shipment_id", "hub_raw", "scan_stage", "scan_time_utc", "recorded_seq"])

    return shipments_df, scans_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Delhi cold-chain logistics dataset.")
    parser.add_argument("--rows", type=int, default=100_000, help="Number of shipment rows (use 1500000 for full scale).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed.")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Output directory for generated files.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shipments_df, scans_df = generate_coldchain_dataset(n_rows=args.rows, seed=args.seed)

    shipments_path = args.output_dir / "shipments_raw.parquet"
    scans_path = args.output_dir / "warehouse_scans_raw.parquet"
    try:
        shipments_df.to_parquet(shipments_path, index=False)
        scans_df.to_parquet(scans_path, index=False)
        print(f"Saved {len(shipments_df):,} shipments -> {shipments_path}")
        print(f"Saved {len(scans_df):,} scan events -> {scans_path}")
    except ImportError:
        shipments_csv = args.output_dir / "shipments_raw.csv"
        scans_csv = args.output_dir / "warehouse_scans_raw.csv"
        shipments_df.to_csv(shipments_csv, index=False)
        scans_df.to_csv(scans_csv, index=False)
        print(f"Saved {len(shipments_df):,} shipments -> {shipments_csv}")
        print(f"Saved {len(scans_df):,} scan events -> {scans_csv}")

    late_rate = shipments_df["is_late"].mean()
    print(f"Class balance (late delivery rate): {late_rate:.2%}")


if __name__ == "__main__":
    main()
