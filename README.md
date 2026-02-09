🧬 Liver Disease Outcome & Survival Prediction – End-to-End ML Pipeline

Predict patient mortality risk, disease stage, and survival time using clinical and laboratory features.
Built with a complete Data Science workflow and deployed through a React + FastAPI stack.

👨‍💻 Team

Ibrahim Raboudi

Anas Kareem

Othmen Chtiui

📌 Description

This project is based on a clinical liver disease dataset containing demographic, laboratory, and histologic variables.

We build three complementary predictive systems:

Binary Classification → Predict patient mortality (Status)

Multi-class Classification → Predict disease stage (Stage)

Survival Analysis → Estimate time-to-event using N_Days

The project follows a structured data science roadmap:
EDA → Preprocessing → Feature Engineering → Modeling → Evaluation → API → Frontend → Deployment

📊 Dataset Overview
| Source                          | Link                                                                                                                                     | Records      | Features                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------------------------------- |
| UCI Machine Learning Repository | [Cirrhosis Patient Survival Prediction Dataset](https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1) | 418 patients | 20 clinical & laboratory variables |


This dataset contains medical records of patients with liver cirrhosis, including demographic data, laboratory test results, clinical findings, and survival outcomes.

Variable Type	Examples
Demographic	Age, Sex
Clinical Signs	Ascites, Hepatomegaly, Edema
Lab Values	Bilirubin, Cholesterol, Albumin, Copper
Blood Tests	SGOT, Platelets, Prothrombin
Target Variables	Status, Stage, N_Days
🎯 Targets

Status

C = Censored

CL = Censored (Liver transplant)

D = Death

Stage

1 → Early stage

4 → Advanced liver disease

Survival

N_Days (Time until death, transplant, or study end)

🚀 Project Objectives
1️⃣ Mortality Risk Prediction (Binary Classification)

Predict whether a patient is at risk of death.

Models:

Logistic Regression

Random Forest

XGBoost / CatBoost

Metrics:

Accuracy

Recall (critical in medical context)

F1-score

ROC-AUC

2️⃣ Disease Stage Prediction (Multi-Class Classification)

Predict histologic stage (1–4) from medical indicators.

Models:

Random Forest

Gradient Boosting

Multi-class XGBoost

Metrics:

Accuracy

Confusion Matrix

Macro F1-score

3️⃣ Survival Analysis (Advanced Modeling)

Estimate survival probability over time using:

Kaplan–Meier Curves

Cox Proportional Hazards Model

This provides:

Survival probability estimation

Risk scoring

Hazard ratios interpretation

🗺 Roadmap
Phase	Deliverable
EDA	Data exploration notebook with distribution plots, missing value analysis
Preprocessing	Clean pipeline (src/preprocess.py)
Feature Engineering	Encoded categorical features, scaled lab variables
Modeling	Tuned ML models + cross-validation
Survival Analysis	Cox model + Kaplan-Meier plots
API (FastAPI)	/predict_status, /predict_stage, /predict_survival endpoints
Frontend (React)	Risk dashboard + survival probability visualization
Deployment	Dockerized stack + cloud deployment
🏗 Tech Stack

Python (pandas, scikit-learn, lifelines, XGBoost)

FastAPI

React

Docker

MLflow (experiment tracking)


📈 Expected Impact

This system helps:

Identify high-risk patients early

Support medical decision-making

Provide survival probability estimation

Assist in treatment prioritization
