# Heart Attack Risk Prediction – ML End-to-End Pipeline

Predict the probability of a heart attack from clinical features and serve results through a React + FastAPI web stack.

## Description
This project uses the *Heart Attack Prediction Dataset* to build a binary-classification model that estimates patient risk.  
We follow a 7-week agile roadmap: EDA → preprocessing → gradient-boosting modelling (tracked with MLflow) → REST API → React dashboard → Docker containerisation → cloud deployment.

## Dataset
| Source | Link | Rows | Features | License |
|--------|------|------|----------|---------|
| Kaggle | [Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/heart-attack-prediction-dataset) | 1 319 | 14 (numeric + categorical) | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) |

**Target:** `HeartDisease` (0 = no heart attack, 1 = heart attack)  
**Key predictors:** age, sex, chest-pain type, cholesterol, fasting blood sugar, max heart rate, etc.

## 7-Week Roadmap
| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Scraping & EDA | Jupyter notebook with univariate + bivariate plots, data-quality report |
| 2 | Pre-processing & Feature Engineering | Clean pipeline (`src/preprocess.py`), engineered features stored in `data/processed/` |
| 3 | Modelling (Gradient Boosting) & MLflow | Tuned CatBoost / XGBoost, registered in MLflow, `metrics.txt` |
| 4 | API Development (FastAPI) | `/predict` endpoint with input validation & automated tests |
| 5 | Frontend Development (React) | Single-page app: form input → probability gauge → risk interpretation |
| 6 | Containerisation (Docker) | `docker-compose up` spins up API + React + Postgres for logs |
| 7 | Deployment & Final Review | App live on Render (or AWS EC2), slides + demo video |

## Project Structure
