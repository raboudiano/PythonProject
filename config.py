"""
Configuration file for the Cirrhosis Prediction Project
"""

# Data paths
DATA_PATH = "Data/cirrhosis.csv"
CLEANED_DATA_PATH = "Data/cirrhosis_cleaned.csv"
REPORTS_PATH = "Reports/"
MODELS_PATH = "Models/"

# EDA settings
EDA_REPORT_PATH = "Reports/Cirrhosis_EDA_Report_Enhanced.pdf"
VISUALIZATION_DPI = 300

# MLflow settings
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
MLFLOW_EXPERIMENT_NAME = "Cirrhosis_Prediction"

# Model configuration
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Target variables
PRIMARY_TARGET = "Status"  # C, CL, D
SECONDARY_TARGETS = ["Stage", "N_Days"]

# Features to exclude
EXCLUDE_COLUMNS = ["ID"]
