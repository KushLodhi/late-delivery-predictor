"""FastAPI inference, SHAP explainability, and drift monitoring service."""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    BatchPredictionResponse,
    BatchShipmentRequest,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
    ShapContribution,
    ShipmentRequest,
)
from src.config import METADATA_PATH, MODEL_PATH, REFERENCE_STATS_PATH
from src.drift import detect_drift, load_reference_stats
from src.explainability import _build_explainer, explain_single_shipment
from src.preprocessing import clean_and_engineer_features, get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("delhicold.api")


class ModelState:
    """In-memory holder for loaded model, metadata, SHAP explainer, and drift buffer."""

    def __init__(self) -> None:
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.threshold: float = 0.25
        self.reference_stats: Dict[str, Any] | None = None
        self.explainer: Any = None
        self.recent_buffer: Deque[Dict[str, Any]] = deque(maxlen=1000)

    def load_artifacts(self) -> None:
        if MODEL_PATH.exists():
            import joblib

            self.model = joblib.load(MODEL_PATH)
            logger.info("Loaded model artifact from %s", MODEL_PATH)
            try:
                self.explainer = _build_explainer(self.model)
                logger.info("Initialized fast SHAP explainer.")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not pre-initialize SHAP explainer: %s", exc)
                self.explainer = None

        if METADATA_PATH.exists():
            self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            self.threshold = float(self.metadata.get("optimal_threshold", 0.25))

        if REFERENCE_STATS_PATH.exists():
            self.reference_stats = load_reference_stats(REFERENCE_STATS_PATH)
            logger.info("Loaded drift reference stats from %s", REFERENCE_STATS_PATH)


state = ModelState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.load_artifacts()
    yield


app = FastAPI(
    title="DelhiCold Logistics — Late Delivery Early-Warning API",
    description=(
        "Predicts late cold-chain deliveries across 12 Delhi/NCR hubs with "
        "cost-matrix thresholding, SHAP explanations, and PSI/KS drift monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_tier_and_action(prob: float, threshold: float) -> tuple[str, str]:
    if prob >= max(threshold + 0.35, 0.70):
        return (
            "CRITICAL",
            "Immediate dispatch escalation: assign priority express reefer slot and alert receiving hub.",
        )
    if prob >= max(threshold + 0.15, 0.45):
        return (
            "HIGH",
            "Trigger early-warning intervention: verify cabin temp, clear dock queue, and notify route supervisor.",
        )
    if prob >= threshold:
        return (
            "MEDIUM",
            "Flag on hub monitor: prioritize gate release and monitor live corridor congestion.",
        )
    return ("LOW", "Proceed with standard cold-chain dispatch schedule.")


def _prepare_dataframe(items: List[ShipmentRequest]) -> pd.DataFrame:
    raw_df = pd.DataFrame([item.model_dump() for item in items])
    return clean_and_engineer_features(raw_df)


def _require_model() -> None:
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not loaded. Run `python -m src.train` first.",
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "model_missing",
        model_loaded=state.model is not None,
        model_family=str(state.metadata.get("best_model_family", "unloaded")),
        decision_threshold=state.threshold,
        reference_stats_loaded=state.reference_stats is not None,
    )


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    if not state.metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")
    return state.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(shipment: ShipmentRequest) -> PredictionResponse:
    _require_model()
    t0 = time.perf_counter()

    clean_df = _prepare_dataframe([shipment])
    feature_cols = get_feature_columns()
    prob = float(state.model.predict_proba(clean_df[feature_cols])[0, 1])

    # Record into sliding drift buffer
    state.recent_buffer.append(clean_df.iloc[0].to_dict())
    if state.reference_stats is not None and len(state.recent_buffer) >= 50 and len(state.recent_buffer) % 50 == 0:
        detect_drift(pd.DataFrame(list(state.recent_buffer)), state.reference_stats, log_warnings=True)

    is_late_alert = prob >= state.threshold
    risk_tier, action = _risk_tier_and_action(prob, state.threshold)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return PredictionResponse(
        shipment_id=shipment.shipment_id or "SHP-LIVE-001",
        origin_hub_id=str(clean_df["origin_hub_id"].iloc[0]),
        dest_hub_id=str(clean_df["dest_hub_id"].iloc[0]),
        late_probability=round(prob, 4),
        decision_threshold=round(state.threshold, 4),
        is_late_alert=is_late_alert,
        risk_tier=risk_tier,
        recommended_action=action,
        latency_ms=latency_ms,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(shipment: ShipmentRequest) -> ExplanationResponse:
    _require_model()
    t0 = time.perf_counter()

    clean_df = _prepare_dataframe([shipment])
    feature_cols = get_feature_columns()
    X_row = clean_df[feature_cols]
    prob = float(state.model.predict_proba(X_row)[0, 1])

    try:
        shap_out = explain_single_shipment(state.model, X_row, explainer=state.explainer, top_k=8)
    except Exception as exc:
        logger.warning("SHAP explanation fallback triggered: %s", exc)
        shap_out = {"base_value": 0.0, "top_contributions": []}

    is_late_alert = prob >= state.threshold
    risk_tier, action = _risk_tier_and_action(prob, state.threshold)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return ExplanationResponse(
        shipment_id=shipment.shipment_id or "SHP-LIVE-001",
        origin_hub_id=str(clean_df["origin_hub_id"].iloc[0]),
        dest_hub_id=str(clean_df["dest_hub_id"].iloc[0]),
        late_probability=round(prob, 4),
        decision_threshold=round(state.threshold, 4),
        is_late_alert=is_late_alert,
        risk_tier=risk_tier,
        recommended_action=action,
        latency_ms=latency_ms,
        base_value=shap_out["base_value"],
        top_contributions=[ShapContribution(**c) for c in shap_out["top_contributions"]],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchShipmentRequest) -> BatchPredictionResponse:
    _require_model()
    t0 = time.perf_counter()

    clean_df = _prepare_dataframe(payload.shipments)
    feature_cols = get_feature_columns()
    probs = state.model.predict_proba(clean_df[feature_cols])[:, 1]

    preds: List[PredictionResponse] = []
    for idx, (req, prob) in enumerate(zip(payload.shipments, probs)):
        p = float(prob)
        tier, action = _risk_tier_and_action(p, state.threshold)
        preds.append(
            PredictionResponse(
                shipment_id=req.shipment_id or f"SHP-BATCH-{idx + 1}",
                origin_hub_id=str(clean_df["origin_hub_id"].iloc[idx]),
                dest_hub_id=str(clean_df["dest_hub_id"].iloc[idx]),
                late_probability=round(p, 4),
                decision_threshold=round(state.threshold, 4),
                is_late_alert=p >= state.threshold,
                risk_tier=tier,
                recommended_action=action,
                latency_ms=0.0,
            )
        )
        state.recent_buffer.append(clean_df.iloc[idx].to_dict())

    drift_summary = None
    if state.reference_stats is not None and len(clean_df) >= 20:
        drift_summary = detect_drift(clean_df, state.reference_stats, log_warnings=True)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return BatchPredictionResponse(
        count=len(preds),
        late_alert_count=sum(1 for p in preds if p.is_late_alert),
        latency_ms=latency_ms,
        predictions=preds,
        drift_summary=drift_summary,
    )


@app.get("/drift/status")
def drift_status() -> Dict[str, Any]:
    if state.reference_stats is None:
        raise HTTPException(status_code=404, detail="Reference drift stats not loaded.")
    if len(state.recent_buffer) == 0:
        return {
            "n_current_rows": 0,
            "overall_status": "NO_TRAFFIC_YET",
            "drifted_features": [],
            "moderate_features": [],
            "features": {},
        }
    return detect_drift(pd.DataFrame(list(state.recent_buffer)), state.reference_stats, log_warnings=True)
