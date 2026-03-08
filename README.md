# 🧬 Cirrhosis Prediction Project

Clean end-to-end ML pipeline for liver cirrhosis data, with a deployable FastAPI service that auto-selects the best model.

## What this project does

- Cleans and preprocesses cirrhosis data
- Trains multiple classification models
- Compares model metrics and selects the best model by `test_f1`
- Serves predictions through a FastAPI API
- Provides Swagger docs and a minimal frontend for quick testing
- Supports local and Docker execution

## Current deployed scope

The current deployed API focuses on **Status prediction** (target: `Status`).

Best model is selected from:
- `Models/training_results.json`

Deployment artifacts are stored in:
- `Models/best_model_bundle.joblib`
- `Models/best_model_metadata.json`

## Project structure (important files)

- `main.py` → full data pipeline runner
- `model_training.py` → model training/comparison logic
- `deploy_model.py` → best-model selection and bundle persistence
- `api.py` → FastAPI app and endpoints
- `frontend/index.html` → minimal UI for manual prediction testing
- `requirements-api.txt` → API runtime dependencies
- `Dockerfile` / `docker-compose.yml` → containerized deployment

## API endpoints

- `GET /health`
- `GET /model/info`
- `GET /model/sample-input`
- `POST /predict`

Swagger UI:
- `http://127.0.0.1:8000/docs`

Frontend:
- `http://127.0.0.1:8000/`

## Local run

```bash
pip install -r requirements-api.txt
uvicorn api:app --reload
```

Then open:
- Swagger: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/`

## Docker run

```bash
docker compose up --build
```

Then open:
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- Frontend: `http://127.0.0.1:8000/`

## How to test (recommended order)

### 1) Test in Swagger first

1. Open `http://127.0.0.1:8000/docs`
2. Execute `GET /health`
3. Execute `GET /model/sample-input`
4. Copy response JSON
5. Execute `POST /predict` with that JSON

Expected result:
- HTTP 200 with predicted class and probabilities

### 2) Test in frontend

1. Open `http://127.0.0.1:8000/`
2. Click **Load Sample**
3. (Optional) edit feature values
4. Click **Predict**
5. Check response panel

## Roadmap

### Phase 1 — Core API (Done)

- Data cleaning + model comparison pipeline
- Best model auto-selection by `test_f1`
- FastAPI endpoints + Swagger docs
- Minimal frontend for quick testing
- Docker setup for deployment

### Phase 2 — Model Quality (Next)

- Add cross-validation summary to API metadata
- Add model versioning in metadata file
- Add confidence threshold handling for uncertain predictions

### Phase 3 — Productization

- Add authentication for protected API usage
- Add request logging and prediction monitoring
- Add CI checks for lint, tests, and container build

### Phase 4 — Feature Expansion

- Add `Stage` prediction endpoint
- Add survival analysis endpoint (`N_Days`)
- Add richer frontend dashboard while keeping current minimal mode

## Notes

- If port `8000` is busy, stop old process or run on another port:

```bash
uvicorn api:app --reload --port 8001
```

- The API validates feature keys strictly. Missing/extra fields return `422`.

## Team

- Ibrahim Raboudi
- Anas Kareem
- Othmen Chtioui
