# DelhiCold Logistics — Late-Delivery Early-Warning ML System

An end-to-end, reproducible machine learning system that predicts late cold-chain deliveries across **12 Delhi/NCR hubs**, explains every prediction with **SHAP**, monitors live feature drift with **PSI & KS tests**, and serves staff via **FastAPI + Streamlit**.

## Project Structure

```text
delhi-coldchain-late-delivery/
├── app/
│   ├── api.py                 # FastAPI service (/health, /predict, /explain, /predict/batch, /drift/status)
│   ├── schemas.py             # Pydantic v2 input validation (accepts canonical & legacy hub codes)
│   └── streamlit_app.py       # Staff early-warning web console (single, batch CSV, drift monitor)
├── notebooks/
│   └── 01_training_and_evaluation.ipynb
├── src/
│   ├── config.py              # 12 hubs, legacy rename map, INR cost matrix, feature lists
│   ├── data_generator.py      # Synthetic 12-hub cold-chain data generator (scales to 1.5M+ rows)
│   ├── preprocessing.py       # UTC normalization, out-of-order scan sorting, hub canonicalization, time split
│   ├── pipeline.py            # Reproducible scikit-learn ColumnTransformer + Pipeline
│   ├── threshold.py           # Cold-chain INR cost-matrix threshold optimizer
│   ├── explainability.py      # Global & individual SHAP explanations
│   ├── drift.py               # PSI + KS-test drift monitor with structured warning logs
│   └── train.py               # 4-model training (LogReg, RF, XGBoost, LightGBM) + RandomizedSearchCV
├── tests/                     # Pytest suite (preprocessing, pipeline, drift, API & <300ms latency)
├── Dockerfile
├── docker-compose.yml         # Single-command fresh-clone startup
├── run.ps1 / run.bat          # Windows 11 native one-command runner
├── MODEL_CARD.md              # Model card & business justification for threshold choice
└── requirements.txt
```

## Quick Start on Windows 11 (One Command)

### Option 1: Docker Desktop (Recommended — Fresh-Clone One-Command Run)
Open **PowerShell** inside the extracted project folder and run:

```powershell
docker compose up --build
```

This single command automatically:
1. Builds the Python 3.11 container with all dependencies.
2. Generates the dataset, trains & tunes all 4 models, computes global SHAP & drift reference stats, and saves `models/late_delivery_model.joblib` (if not already present).
3. Starts the **FastAPI API** at `http://localhost:8000/docs`.
4. Starts the **Streamlit Staff Console** at `http://localhost:8501`.

To train on the full **1.5 million rows** in Docker:
```powershell
$env:TRAIN_ROWS="1500000"; docker compose up --build
```

### Option 2: Native Windows 11 Python (PowerShell)
```powershell
.\run.ps1
```
Optional flags:
* Full 1.5M-row training run: `.\run.ps1 -Rows 1500000`
* Also execute the training notebook in-place (populates all outputs/charts for submission): `.\run.ps1 -RunNotebook`

## Running Unit Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```
Or inside Docker:
```powershell
docker compose run --rm api pytest -v
```

## Executing the Training Notebook Before Submission
To populate all cell outputs and plots inside `notebooks/01_training_and_evaluation.ipynb`:
```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks\01_training_and_evaluation.ipynb
```

## API Endpoints
* `GET /health` — Service health, loaded model family, and active decision threshold.
* `POST /predict` — Single shipment risk prediction (`< 300ms`).
* `POST /explain` — Single shipment prediction + top individual SHAP feature contributions.
* `POST /predict/batch` — Batch scoring (up to 500 shipments) + batch drift check.
* `GET /drift/status` — Live PSI & KS-test feature drift report against training distributions.
* `GET /model/info` — Full training leaderboard, cost-matrix metrics, and global SHAP ranking.

## Pushing to GitHub (Windows 11 PowerShell)
After running `docker compose up --build` (or `.\run.ps1 -RunNotebook`) so `models/late_delivery_model.joblib` and the executed notebook are populated:

```powershell
git init
git add .
git commit -m "Initial commit: DelhiCold late-delivery ML model, API, Streamlit UI, and Docker setup"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

See [MODEL_CARD.md](MODEL_CARD.md) for the full business justification of the cost-matrix decision threshold.
