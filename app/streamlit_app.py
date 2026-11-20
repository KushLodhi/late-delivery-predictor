"""Streamlit Early-Warning Operations Screen for DelhiCold Logistics Staff."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import CANONICAL_HUBS, CARGO_TYPES, LEGACY_HUB_MAP, VEHICLE_TYPES, WEATHER_CONDITIONS

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(
    page_title="DelhiCold Early-Warning Console",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ DelhiCold Logistics — Late-Delivery Early-Warning Console")
st.caption("Operations decision-support screen across 12 Delhi/NCR cold-chain distribution hubs")

# Sidebar API health check
with st.sidebar:
    st.header("System Status")
    try:
        health = httpx.get(f"{API_URL}/health", timeout=5.0).json()
        if health.get("model_loaded"):
            st.success(f"API Online | Model: `{health['model_family']}`")
            st.metric("Cost-Optimal Threshold", f"{health['decision_threshold']:.2f}")
        else:
            st.warning("API online, but model artifact is missing.")
    except Exception as exc:
        st.error(f"Cannot reach API at `{API_URL}`\n\n{exc}")

    st.divider()
    st.markdown("**Business Cost Matrix (INR)**")
    st.markdown("- Missed late delivery (FN): **₹5,000**")
    st.markdown("- Early intervention (FP / TP): **₹400**")
    st.markdown("- On-time normal (TN): **₹0**")

tab_single, tab_batch, tab_drift = st.tabs(
    ["🚛 Single Shipment Check", "📦 Batch Manifest Upload", "📡 Feature Drift Monitor"]
)

hub_options = list(CANONICAL_HUBS.keys())
hub_labels = {k: f"{k} — {v['name']}" for k, v in CANONICAL_HUBS.items()}

with tab_single:
    st.subheader("Check Shipment Late-Delivery Risk")
    col1, col2, col3 = st.columns(3)

    with col1:
        shipment_id = st.text_input("Shipment ID", value="SHP-DEL-2025-8841")
        origin_hub = st.selectbox("Origin Hub", hub_options, format_func=lambda x: hub_labels[x], index=0)
        dest_hub = st.selectbox("Destination Hub", hub_options, format_func=lambda x: hub_labels[x], index=8)
        cargo_type = st.selectbox("Cargo Type", CARGO_TYPES, index=0)
        vehicle_type = st.selectbox("Vehicle Type", VEHICLE_TYPES, index=1)

    with col2:
        weather_condition = st.selectbox("Weather Condition", WEATHER_CONDITIONS, index=2)
        distance_km = st.number_input("Route Distance (km)", min_value=1.0, max_value=300.0, value=42.5, step=1.0)
        planned_transit_hours = st.number_input("Planned Transit (hours)", min_value=0.5, max_value=24.0, value=2.4, step=0.1)
        warehouse_dwell_hours = st.number_input("Warehouse Dwell (hours)", min_value=0.0, max_value=24.0, value=2.6, step=0.1)
        route_congestion_index = st.slider("Route Congestion Index", min_value=0.0, max_value=1.0, value=0.82, step=0.01)

    with col3:
        default_ist = datetime.now(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")
        dispatch_time_raw = st.text_input("Dispatch Timestamp (IST / UTC)", value=default_ist)
        temp_setpoint_c = st.number_input("Temp Setpoint (°C)", min_value=-30.0, max_value=20.0, value=4.0, step=0.5)
        avg_cabin_temp_c = st.number_input("Avg Cabin Temp (°C)", min_value=-30.0, max_value=35.0, value=6.8, step=0.5)
        vehicle_age_years = st.number_input("Vehicle Age (years)", min_value=0.1, max_value=20.0, value=5.5, step=0.5)
        payload_weight_kg = st.number_input("Payload Weight (kg)", min_value=100.0, max_value=15000.0, value=2100.0, step=50.0)
        out_of_order_scan_flag = st.checkbox("Out-of-order warehouse scan detected", value=False)

    if st.button("Predict & Explain Risk", type="primary", use_container_width=True):
        payload = {
            "shipment_id": shipment_id,
            "origin_hub_raw": origin_hub,
            "dest_hub_raw": dest_hub,
            "dispatch_time_raw": dispatch_time_raw,
            "source_timezone": "Asia/Kolkata",
            "cargo_type": cargo_type,
            "vehicle_type": vehicle_type,
            "weather_condition": weather_condition,
            "distance_km": distance_km,
            "planned_transit_hours": planned_transit_hours,
            "warehouse_dwell_hours": warehouse_dwell_hours,
            "scan_count": 5,
            "out_of_order_scan_flag": int(out_of_order_scan_flag),
            "temp_setpoint_c": temp_setpoint_c,
            "avg_cabin_temp_c": avg_cabin_temp_c,
            "vehicle_age_years": vehicle_age_years,
            "payload_weight_kg": payload_weight_kg,
            "route_congestion_index": route_congestion_index,
        }
        try:
            resp = httpx.post(f"{API_URL}/explain", json=payload, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Late Probability", f"{data['late_probability'] * 100:.1f}%")
            m2.metric("Decision Threshold", f"{data['decision_threshold'] * 100:.1f}%")
            m3.metric("Risk Tier", data["risk_tier"])
            m4.metric("API Latency", f"{data['latency_ms']:.1f} ms")

            if data["is_late_alert"]:
                st.error(f"⚠️ **EARLY WARNING — LATE DELIVERY RISK**: {data['recommended_action']}")
            else:
                st.success(f"✅ **ON-TIME EXPECTED**: {data['recommended_action']}")

            contribs = data.get("top_contributions", [])
            if contribs:
                st.markdown("#### Individual SHAP Explanation (Why this prediction?)")
                cdf = pd.DataFrame(contribs)
                fig, ax = plt.subplots(figsize=(8.0, 3.8))
                colors = ["#D9381E" if v > 0 else "#1B7A43" for v in cdf["shap_value"][::-1]]
                ax.barh(cdf["feature"][::-1], cdf["shap_value"][::-1], color=colors)
                ax.axvline(0, color="#333333", linewidth=0.8)
                ax.set_xlabel("SHAP Contribution (Red = Increases Late Risk, Green = Reduces Risk)")
                ax.grid(axis="x", linestyle="--", alpha=0.4)
                fig.tight_layout()
                st.pyplot(fig)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with tab_batch:
    st.subheader("Batch Shipment Scoring (CSV Upload)")
    st.caption(
        "Upload a CSV containing shipment columns (`origin_hub_raw`, `dest_hub_raw`, `dispatch_time_raw`, "
        "`cargo_type`, `vehicle_type`, `weather_condition`, `distance_km`, `planned_transit_hours`, etc.)."
    )
    uploaded = st.file_uploader("Upload Manifest CSV", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded).head(500)
        st.dataframe(df_in.head(10), use_container_width=True)
        if st.button("Score Batch via API", type="primary"):
            records = df_in.where(pd.notnull(df_in), None).to_dict(orient="records")
            try:
                resp = httpx.post(f"{API_URL}/predict/batch", json={"shipments": records}, timeout=20.0)
                resp.raise_for_status()
                out = resp.json()
                st.success(
                    f"Scored {out['count']} shipments in {out['latency_ms']:.1f} ms — "
                    f"**{out['late_alert_count']} early-warning alerts** triggered."
                )
                pred_df = pd.DataFrame(out["predictions"])
                st.dataframe(pred_df, use_container_width=True)
                st.download_button(
                    "Download Scored Manifest CSV",
                    data=pred_df.to_csv(index=False).encode("utf-8"),
                    file_name="delhicold_scored_manifest.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Batch scoring failed: {exc}")

with tab_drift:
    st.subheader("Live Feature Drift Monitor (PSI & KS-Test)")
    st.caption("Compares recent incoming API traffic against training reference distributions.")
    if st.button("Refresh Drift Status"):
        try:
            drift = httpx.get(f"{API_URL}/drift/status", timeout=5.0).json()
            status = drift.get("overall_status", "UNKNOWN")
            if status == "DRIFT_WARNING":
                st.error(f"🚨 Drift Warning! Shifted features: {', '.join(drift.get('drifted_features', []))}")
            elif status == "MONITOR":
                st.warning(f"⚠️ Moderate shift detected: {', '.join(drift.get('moderate_features', []))}")
            elif status == "NO_TRAFFIC_YET":
                st.info("No live prediction traffic recorded yet. Run predictions first.")
            else:
                st.success("✅ Feature distributions are stable (HEALTHY).")

            if drift.get("features"):
                st.dataframe(pd.DataFrame(drift["features"]).T, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not fetch drift status: {exc}")
