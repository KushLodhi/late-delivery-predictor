# Model Card — DelhiCold Logistics Late-Delivery Early-Warning Model

## 1. Model Overview
* **System Name:** DelhiCold Late-Delivery Early-Warning Classifier (`v1.0.0`)
* **Owner / Intended Users:** Dispatch supervisors and hub operations staff at **DelhiCold Logistics** across 12 Delhi/NCR cold-chain hubs.
* **Primary Task:** Binary classification at dispatch time (`is_late` = 1 if actual delivery exceeds planned SLA timestamp, else 0), outputting calibrated risk probabilities, risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), and per-shipment SHAP explanations.
* **Model Family & Pipeline:** Reproducible `scikit-learn` `Pipeline` (`ColumnTransformer` with median/mode imputation, optional `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`) paired with the top-performing gradient-boosted tree classifier selected across four trained families (Logistic Regression, Random Forest, XGBoost, LightGBM) and tuned via `RandomizedSearchCV` with `TimeSeriesSplit`.

## 2. Intended Use & Out-of-Scope Uses
* **Intended Use:** Triggering proactive cold-chain dispatch interventions (priority gate release, express reefer reassignment, receiving-hub alert, cabin temperature verification) *before* a shipment leaves the origin hub.
* **Out-of-Scope Uses:** Automated driver penalization, payroll deductions, or replacing mandatory regulatory cold-chain temperature telemetry for pharmaceuticals/vaccines.

## 3. Data Sources & Known Data Quality Hazards Addressed
The pipeline is engineered for shipment, route, vehicle, and warehouse scan records across 12 Delhi/NCR hubs (`HUB-DEL-OKH`, `HUB-DEL-AZD`, `HUB-DEL-GHZ`, `HUB-DEL-BWN`, `HUB-DEL-NJF`, `HUB-DEL-ALP`, `HUB-HR-KND`, `HUB-HR-MNS`, `HUB-HR-GGN`, `HUB-HR-FBD`, `HUB-UP-NOI`, `HUB-UP-GZB`), scaling from 100,000 rows (fast local verification) up to 1,500,000+ rows (`--rows 1500000`).

| Data Quality Issue | Root Cause | Implemented Fix (`src/preprocessing.py`) |
| :--- | :--- | :--- |
| **Mixed time zones** | Hub scanners emit `+05:30` (IST), `UTC Z`, naive local IST strings, and occasional `+04:00` misconfigured offsets. | `parse_mixed_timestamp_to_utc()` normalizes all explicit offsets to UTC and localizes naive strings as `Asia/Kolkata` before UTC conversion. |
| **Out-of-order warehouse scans** | Network retries cause warehouse gate/dock scans to arrive out of chronological order. | `reconcile_warehouse_scans()` sorts events by UTC timestamp per `shipment_id`, recovers true dwell hours, and records `out_of_order_scan_flag`. |
| **Post-network-change hub renames** | Legacy hub codes (`DEL-OKH-OLD`, `GGN-HUB-1`, `NOI-SEC63-LEGACY`, `FBD-OLD`, etc.) were renamed on `2025-07-01`. | `canonicalize_hub_name()` maps all historical and live legacy codes to the 12 canonical hub IDs and adds `post_network_change`. |
| **Temporal leakage** | Random train/test splits leak future seasonal and congestion patterns into training. | `chronological_split()` enforces a strict 70% train / 15% validation / 15% test split ordered by `dispatch_time_utc`, paired with `TimeSeriesSplit` in CV. |
| **Class imbalance (~20% late)** | Standard accuracy rewards predicting "on-time" for every shipment. | `class_weight="balanced"` (LogReg/RF/LightGBM) and `scale_pos_weight = n_neg / n_pos` (XGBoost), evaluated via PR-AUC (`average_precision`) and business cost. |

## 4. Business Justification for Decision Threshold Choice
A default probability threshold of **0.50** implicitly assumes that missing a late delivery costs the same as issuing a false alarm. In cold-chain logistics, those costs are sharply asymmetric:

| Outcome | Operational Meaning | Agreed Unit Cost (INR) |
| :--- | :--- | ---: |
| **False Negative (FN)** | Late shipment missed -> thermal excursion/spoilage claim, SLA penalty, customer churn | **₹5,000** |
| **False Positive (FP)** | On-time shipment flagged -> unnecessary priority dispatch / ops escalation call | **₹400** |
| **True Positive (TP)** | Late shipment caught early -> proactive intervention mitigates spoilage (saves ₹4,600 net) | **₹400** |
| **True Negative (TN)** | On-time shipment proceeds normally | **₹0** |

### Why We Lower the Threshold Below 0.50
Let $p$ be the predicted probability that a shipment will be late:
* **Expected cost if we trigger an alert:** $E[C \mid \text{alert}] = p \cdot 400 + (1 - p) \cdot 400 = \text{₹400}$
* **Expected cost if we do nothing:** $E[C \mid \text{no alert}] = p \cdot 5000 + (1 - p) \cdot 0 = 5000p$

Triggering an early-warning intervention is financially superior whenever:
$$400 \le 5000p \implies p^* = \frac{C_{FP}}{C_{FP} + (C_{FN} - C_{TP})} = \frac{400}{400 + 4600} = 0.08$$

Because class-weighted models produce shifted (uncalibrated) probabilities relative to raw base rates, `src/threshold.py` performs an **empirical validation grid search** across `[0.05, 0.95]` to select the exact threshold that minimizes realized total INR cost on the chronological validation split (persisted in `models/model_metadata.json`).

**Executive talking point:** Every prevented late cold-chain delivery saves **₹4,600 net** (`₹5,000 - ₹400`), meaning we can afford up to **11.5 false-alarm interventions** (`₹4,600 / ₹400`) to catch a single missed late delivery and still break even. Lowering the threshold from `0.50` to the cost-optimal operating point trades a modest increase in low-cost dispatch checks for a large reduction in high-cost cold-chain spoilage and SLA penalties.

## 5. Evaluation & Success Criteria
Running `python -m src.train` writes the exact observed leaderboard, PR-AUC lift over baseline, test-set confusion matrix, and INR cost comparison into `models/model_metadata.json`.
* **Baseline Beat Target:** >= 20% relative improvement over the stratified baseline on PR-AUC (`meets_20pct_lift_target: true`).
* **Latency SLA:** FastAPI `/predict` responds well under **300 ms** (verified by `tests/test_api.py`).
* **Reproducibility:** Fresh-clone single-command execution via `docker compose up --build` or `.\run.ps1`.

## 6. Explainability (SHAP)
* **Global Explainability:** Computed via `shap.TreeExplainer` on held-out test records (`models/shap_global.json` and `models/shap_global_bar.png`). Top drivers typically include `warehouse_dwell_hours`, `route_congestion_index`, `dwell_to_transit_ratio`, `weather_condition` (monsoon/fog), `temp_deviation_c`, and `is_peak_hour`.
* **Individual Explainability:** Exposed in real time via `POST /explain` and rendered as a color-coded horizontal bar chart in the Streamlit console so dispatch staff see *why* a specific shipment triggered an alert (e.g., high dwell time + heavy rain + cabin temperature drift).

## 7. Drift Monitoring & Governance
* `src/drift.py` captures quantile bin edges, proportions, and reference samples from the training split (`models/reference_stats.json`).
* Live API traffic is buffered and compared against training distributions using **Population Stability Index (PSI)** and two-sample **Kolmogorov-Smirnov (KS)** tests.
* **Alert Policy:** Features with `PSI >= 0.20` (or `KS p < 0.01` with `PSI >= 0.10`) log structured `WARNING` events (`delhicold.drift`) and surface in `GET /drift/status` and the Streamlit Drift Monitor tab.
* **Seasonal Caveat:** Delhi experiences sharp seasonal weather shifts (winter fog, summer heatwaves, July–September monsoon). Expected seasonal shifts in `weather_condition` should be monitored alongside model calibration, with quarterly retraining recommended.
