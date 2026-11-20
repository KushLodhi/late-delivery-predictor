"""Pydantic v2 request/response schemas with strict validation for DelhiCold API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from src.config import CANONICAL_HUBS, CARGO_TYPES, LEGACY_HUB_MAP, VEHICLE_TYPES, WEATHER_CONDITIONS
from src.preprocessing import canonicalize_hub_name

VALID_HUB_CODES = set(CANONICAL_HUBS.keys()) | set(LEGACY_HUB_MAP.keys())


class ShipmentRequest(BaseModel):
    """Input payload for a single cold-chain shipment late-delivery prediction."""

    shipment_id: Optional[str] = Field(default="SHP-LIVE-001", max_length=64)
    origin_hub_raw: str = Field(..., description="Origin hub code (canonical or pre-rename legacy code).")
    dest_hub_raw: str = Field(..., description="Destination hub code (canonical or pre-rename legacy code).")
    dispatch_time_raw: str = Field(
        ...,
        description="Dispatch timestamp in ISO-8601 (+05:30, UTC Z, +04:00) or naive IST format.",
        examples=["2025-11-20T09:15:00+05:30"],
    )
    source_timezone: str = Field(default="Asia/Kolkata", max_length=32)
    cargo_type: Literal[
        "vaccines_pharma",
        "fresh_dairy",
        "frozen_meat",
        "fresh_produce",
        "ice_cream_confectionery",
    ]
    vehicle_type: Literal[
        "reefer_van_1t",
        "reefer_truck_3t",
        "reefer_truck_7t",
        "insulated_lcv",
    ]
    weather_condition: Literal[
        "clear",
        "haze_smog",
        "heavy_rain_monsoon",
        "extreme_heatwave",
        "dense_fog",
    ]
    distance_km: float = Field(..., gt=0.0, le=500.0)
    planned_transit_hours: float = Field(..., gt=0.25, le=48.0)
    warehouse_dwell_hours: Optional[float] = Field(default=1.2, ge=0.0, le=48.0)
    scan_count: int = Field(default=4, ge=1, le=50)
    out_of_order_scan_flag: int = Field(default=0, ge=0, le=1)
    temp_setpoint_c: float = Field(default=4.0, ge=-35.0, le=25.0)
    avg_cabin_temp_c: Optional[float] = Field(default=4.5, ge=-40.0, le=45.0)
    vehicle_age_years: Optional[float] = Field(default=3.0, ge=0.0, le=25.0)
    payload_weight_kg: Optional[float] = Field(default=1500.0, gt=0.0, le=25000.0)
    route_congestion_index: Optional[float] = Field(default=0.65, ge=0.0, le=1.0)

    @field_validator("origin_hub_raw", "dest_hub_raw")
    @classmethod
    def validate_hub(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned not in VALID_HUB_CODES:
            raise ValueError(
                f"Unknown hub '{value}'. Expected one of the 12 canonical Delhi/NCR hubs or known legacy codes."
            )
        return canonicalize_hub_name(cleaned)


class BatchShipmentRequest(BaseModel):
    shipments: List[ShipmentRequest] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    shipment_id: str
    origin_hub_id: str
    dest_hub_id: str
    late_probability: float
    decision_threshold: float
    is_late_alert: bool
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str
    latency_ms: float


class ShapContribution(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_late_risk", "decreases_late_risk"]


class ExplanationResponse(PredictionResponse):
    base_value: float
    top_contributions: List[ShapContribution]


class BatchPredictionResponse(BaseModel):
    count: int
    late_alert_count: int
    latency_ms: float
    predictions: List[PredictionResponse]
    drift_summary: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_family: str
    decision_threshold: float
    reference_stats_loaded: bool
