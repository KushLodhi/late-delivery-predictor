"""Central configuration for DelhiCold Logistics late-delivery ML system."""

from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

MODEL_PATH = MODELS_DIR / "late_delivery_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REFERENCE_STATS_PATH = MODELS_DIR / "reference_stats.json"
SHAP_GLOBAL_PATH = MODELS_DIR / "shap_global.json"

# Network change date when legacy hub codes were renamed to canonical IDs
HUB_RENAME_CUTOFF_UTC = "2025-07-01T00:00:00Z"

# 12 Delhi/NCR cold-chain distribution hubs (canonical ID -> metadata)
CANONICAL_HUBS: Dict[str, Dict[str, object]] = {
    "HUB-DEL-OKH": {"name": "Okhla Industrial Cold Hub", "zone": "South Delhi", "congestion_base": 0.72, "lat": 28.5355, "lon": 77.2710},
    "HUB-DEL-AZD": {"name": "Azadpur Mandi Cold Terminal", "zone": "North Delhi", "congestion_base": 0.85, "lat": 28.7150, "lon": 77.1720},
    "HUB-DEL-GHZ": {"name": "Ghazipur Dairy & Pharma Hub", "zone": "East Delhi", "congestion_base": 0.76, "lat": 28.6250, "lon": 77.3260},
    "HUB-DEL-BWN": {"name": "Bawana Logistics Park", "zone": "North-West Delhi", "congestion_base": 0.58, "lat": 28.7980, "lon": 77.0430},
    "HUB-DEL-NJF": {"name": "Najafgarh Agro Cold Hub", "zone": "South-West Delhi", "congestion_base": 0.62, "lat": 28.6090, "lon": 76.9800},
    "HUB-DEL-ALP": {"name": "Alipur NH-44 Cross-Dock", "zone": "Outer North Delhi", "congestion_base": 0.65, "lat": 28.7990, "lon": 77.1330},
    "HUB-HR-KND": {"name": "Kundli Cold-Chain Gateway", "zone": "Sonipat Border", "congestion_base": 0.54, "lat": 28.8850, "lon": 77.1180},
    "HUB-HR-MNS": {"name": "IMT Manesar Reefer Hub", "zone": "Manesar", "congestion_base": 0.60, "lat": 28.3560, "lon": 76.9370},
    "HUB-HR-GGN": {"name": "Gurugram Udyog Vihar Hub", "zone": "Gurugram", "congestion_base": 0.78, "lat": 28.5020, "lon": 77.0810},
    "HUB-HR-FBD": {"name": "Faridabad NIT Cold Depot", "zone": "Faridabad", "congestion_base": 0.64, "lat": 28.4089, "lon": 77.3178},
    "HUB-UP-NOI": {"name": "Noida Sector 63 Cold Hub", "zone": "Noida", "congestion_base": 0.70, "lat": 28.6279, "lon": 77.3812},
    "HUB-UP-GZB": {"name": "Ghaziabad Sahibabad Hub", "zone": "Ghaziabad", "congestion_base": 0.74, "lat": 28.6780, "lon": 77.3670},
}

# Mapping from pre-network-change legacy hub names to canonical IDs
LEGACY_HUB_MAP: Dict[str, str] = {
    "DEL-OKH-OLD": "HUB-DEL-OKH",
    "OKHLA_COLD_01": "HUB-DEL-OKH",
    "AZADPUR-MKT": "HUB-DEL-AZD",
    "GZP-EAST-HUB": "HUB-DEL-GHZ",
    "BWN-NW-01": "HUB-DEL-BWN",
    "NJF-SW-DEPOT": "HUB-DEL-NJF",
    "ALP-GT-KARNAL": "HUB-DEL-ALP",
    "KUNDLI-BORDER": "HUB-HR-KND",
    "MNS-IMT-LEGACY": "HUB-HR-MNS",
    "GGN-HUB-1": "HUB-HR-GGN",
    "FBD-OLD": "HUB-HR-FBD",
    "NOI-SEC63-LEGACY": "HUB-UP-NOI",
    "GZB-SAHIBABAD-01": "HUB-UP-GZB",
}

CARGO_TYPES: List[str] = [
    "vaccines_pharma",
    "fresh_dairy",
    "frozen_meat",
    "fresh_produce",
    "ice_cream_confectionery",
]

VEHICLE_TYPES: List[str] = [
    "reefer_van_1t",
    "reefer_truck_3t",
    "reefer_truck_7t",
    "insulated_lcv",
]

WEATHER_CONDITIONS: List[str] = [
    "clear",
    "haze_smog",
    "heavy_rain_monsoon",
    "extreme_heatwave",
    "dense_fog",
]

# Business Cost Matrix (INR) agreed with DelhiCold Logistics operations
# FN: Missed late cold-chain shipment -> spoilage claim, SLA penalty, customer churn
# FP: Unnecessary early-warning intervention -> priority reroute / ops escalation call
# TP: Late shipment caught early -> intervention cost only (saves FN - TP = INR 4,600 net)
# TN: On-time shipment predicted on-time -> INR 0
COST_MATRIX_INR: Dict[str, float] = {
    "cost_fn": 5000.0,
    "cost_fp": 400.0,
    "cost_tp": 400.0,
    "cost_tn": 0.0,
}

# Numeric and categorical feature lists consumed by the sklearn ColumnTransformer
NUMERIC_FEATURES: List[str] = [
    "distance_km",
    "planned_transit_hours",
    "warehouse_dwell_hours",
    "scan_count",
    "out_of_order_scan_flag",
    "temp_setpoint_c",
    "avg_cabin_temp_c",
    "temp_deviation_c",
    "vehicle_age_years",
    "payload_weight_kg",
    "route_congestion_index",
    "dispatch_hour_ist",
    "dispatch_dayofweek",
    "is_weekend",
    "is_peak_hour",
    "is_monsoon_season",
    "planned_speed_kmph",
    "dwell_to_transit_ratio",
    "origin_hub_congestion_base",
    "dest_hub_congestion_base",
    "post_network_change",
]

CATEGORICAL_FEATURES: List[str] = [
    "origin_hub_id",
    "dest_hub_id",
    "cargo_type",
    "vehicle_type",
    "weather_condition",
    "source_timezone",
]

RANDOM_STATE = 42
