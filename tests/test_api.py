"""Unit tests for FastAPI endpoints, Pydantic validation, legacy hub handling, and latency (<300ms)."""

from __future__ import annotations

import time
import numpy as np
from fastapi.testclient import TestClient

from app.api import app, state
from src.data_generator import generate_coldchain_dataset
from src.drift import compute_reference_stats
from src.preprocessing import clean_and_engineer_features


class _DummyProbModel:
    """Fast deterministic mock model for isolated API contract & latency testing."""

    def predict_proba(self, X):
        # Higher dwell or congestion yields higher late probability
        dwell = X["warehouse_dwell_hours"].fillna(1.0).to_numpy(dtype=float)
        cong = X["route_congestion_index"].fillna(0.5).to_numpy(dtype=float)
        score = np.clip(0.15 + 0.12 * dwell + 0.35 * cong, 0.02, 0.98)
        return np.column_stack([1.0 - score, score])


def _setup_mock_state():
    raw_df, scans_df = generate_coldchain_dataset(n_rows=600, seed=5)
    clean_df = clean_and_engineer_features(raw_df, scans_df)
    state.model = _DummyProbModel()
    state.metadata = {"best_model_family": "mock_lightgbm", "optimal_threshold": 0.25}
    state.threshold = 0.25
    state.reference_stats = compute_reference_stats(clean_df)
    state.explainer = None
    state.recent_buffer.clear()


SAMPLE_PAYLOAD = {
    "shipment_id": "SHP-TEST-001",
    "origin_hub_raw": "DEL-OKH-OLD",  # Legacy hub code must canonicalize to HUB-DEL-OKH
    "dest_hub_raw": "HUB-HR-GGN",
    "dispatch_time_raw": "2025-09-14T09:15:00+05:30",
    "source_timezone": "Asia/Kolkata",
    "cargo_type": "vaccines_pharma",
    "vehicle_type": "reefer_truck_3t",
    "weather_condition": "heavy_rain_monsoon",
    "distance_km": 38.5,
    "planned_transit_hours": 2.2,
    "warehouse_dwell_hours": 2.8,
    "scan_count": 5,
    "out_of_order_scan_flag": 1,
    "temp_setpoint_c": 4.0,
    "avg_cabin_temp_c": 6.4,
    "vehicle_age_years": 5.0,
    "payload_weight_kg": 1850.0,
    "route_congestion_index": 0.85,
}


def test_health_and_predict_under_300ms():
    _setup_mock_state()
    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["model_loaded"] is True

    t0 = time.perf_counter()
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert resp.status_code == 200
    body = resp.json()
    assert body["origin_hub_id"] == "HUB-DEL-OKH"
    assert body["dest_hub_id"] == "HUB-HR-GGN"
    assert 0.0 <= body["late_probability"] <= 1.0
    assert body["is_late_alert"] is True
    assert elapsed_ms < 300.0, f"API latency exceeded 300ms: {elapsed_ms:.2f}ms"


def test_input_validation_rejects_invalid_hub_and_negative_distance():
    _setup_mock_state()
    client = TestClient(app)

    bad_hub = {**SAMPLE_PAYLOAD, "origin_hub_raw": "MUMBAI-FAKE-HUB"}
    assert client.post("/predict", json=bad_hub).status_code == 422

    bad_dist = {**SAMPLE_PAYLOAD, "distance_km": -15.0}
    assert client.post("/predict", json=bad_dist).status_code == 422


def test_batch_predict_and_drift_status_endpoint():
    _setup_mock_state()
    client = TestClient(app)

    batch_payload = {"shipments": [{**SAMPLE_PAYLOAD, "shipment_id": f"SHP-B-{i}"} for i in range(25)]}
    resp = client.post("/predict/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 25
    assert data["drift_summary"] is not None

    drift_resp = client.get("/drift/status")
    assert drift_resp.status_code == 200
    assert drift_resp.json()["n_current_rows"] == 25
