# Cirrhosis Prediction - Complete ML Pipeline

## Project Summary

A comprehensive end-to-end machine learning project for predicting patient outcomes in cirrhosis disease. This project includes enhanced EDA, data cleaning, model training with MLflow, and interactive visualizations.

## Pipeline Stages

### 1. **PHASE 1: Exploratory Data Analysis (EDA)**

**Dataset Insights:**
- **Total Records:** 418 patients
- **Features:** 20 clinical and laboratory variables
- **Missing Values:** Identified and documented
  - Triglycerides: 32.5% missing
  - Cholesterol: 32.1% missing
  - Copper: 25.8% missing
  - Drug, Spiders, Hepatomegaly, Ascites, Alk_Phos, SGOT: ~25% missing
  - Others: <3% missing

**Output:**
- `Reports/Cirrhosis_EDA_Report_Enhanced.pdf` - Comprehensive EDA report including:
  - Dataset overview and statistics
  - Missing data analysis with visualizations
  - Distribution of numeric features
  - Box plots for outlier detection
  - Categorical feature analysis
  - Correlation heatmap
  - Descriptive statistics table

---

### 2. **PHASE 2: Data Cleaning & Preprocessing**

**Cleaning Pipeline:**
1. **Duplicate Removal:** 0 duplicates found
2. **Missing Value Handling:**
   - Numeric: Median imputation
   - Categorical: Most frequent value imputation
3. **Data Inconsistencies:** Removed rows with invalid negative values
4. **Categorical Encoding:** Label encoding for 6 categorical columns
5. **Column Removal:** Removed ID column (non-predictive)
6. **Outlier Detection:** IQR method (multiplier=1.5)

**Results:**
- Original shape: (418, 20)
- Cleaned shape: (163, 19)
- **Rows removed:** 255 (61.2% - aggressive outlier removal)

**Output:**
- `Data/cirrhosis_cleaned.csv` - Cleaned dataset ready for modeling

---

### 3. **PHASE 3: Model Training with MLflow**

**Models Trained:**
1. Logistic Regression
2. Random Forest (100 trees)
3. Gradient Boosting (100 estimators)
4. XGBoost (100 estimators)
5. LightGBM (100 estimators)
6. SVM (RBF kernel)

**Training Configuration:**
- Train/Test split: 80/20
- Stratified split to maintain class distribution
- Standard scaling applied to features
- MLflow tracking for experiment management
- Train size: 130 samples
- Test size: 33 samples

**Output:**
- MLflow tracking database: `mlruns.db`
- `Models/training_results.json` - All metrics in JSON format

---

### 4. **PHASE 4: Model Evaluation & Comparison**

**Performance Ranking (by F1 Score):**

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|-------|---|---|---|---|---|
| **Logistic Regression** | 0.8846 | **0.7879** | **0.7779** | **0.7879** | **0.7806** |
| SVM | 0.8769 | 0.6970 | 0.6633 | 0.6970 | 0.6788 |
| Random Forest | 1.0000 | 0.6667 | 0.6287 | 0.6667 | 0.6465 |
| LightGBM | 1.0000 | 0.6364 | 0.6696 | 0.6364 | 0.6465 |
| XGBoost | 1.0000 | 0.6061 | 0.6564 | 0.6061 | 0.6213 |
| Gradient Boosting | 1.0000 | 0.6061 | 0.6347 | 0.6061 | 0.6184 |

**Key Findings:**
- **Best Model:** Logistic Regression (F1: 0.7806, Accuracy: 0.7879)
- **Overfitting Detection:** Tree-based models show 100% training accuracy but lower test accuracy
- **Logistic Regression Benefits:** 
  - Best generalization (minimal overfitting gap)
  - Interpretable coefficients
  - Computationally efficient
  - Consistent performance across metrics

---

### 5. **PHASE 5: Visualizations Generated**

**PDF Reports:**
- `Reports/Model_Comparison.pdf`
  - Model accuracy comparison charts
  - Training results summary (4-panel layout)
  - ROC curves for model comparison
  - Confusion matrices for each model
  - Feature importance plots for tree-based models

**Interactive HTML Visualizations:**
- `Reports/Model_Comparison_Interactive.html`
  - Interactive bar charts with hover tooltips
  - Filterable metrics display
- `Reports/Model_Radar_Chart.html`
  - Radar/spider chart showing multi-metric performance
  - Easy comparison of all models across dimensions

---

## Project Structure

```
PythonProject/
├── README.md
├── config.py                          # Configuration parameters
├── data_utils.py                      # Data loading and analysis utilities
├── eda_visualization.py               # EDA visualization functions
├── data_cleaning.py                   # Data cleaning pipeline
├── model_training.py                  # Model training with MLflow
├── model_visualization.py             # Visualization utilities
├── main.py                            # Main execution script
│
├── Data/
│   ├── cirrhosis.csv                  # Original dataset (418 rows)
│   └── cirrhosis_cleaned.csv          # Cleaned dataset (163 rows)
│
├── Reports/
│   ├── Cirrhosis_EDA_Report_Enhanced.pdf
│   ├── Model_Comparison.pdf
│   ├── Model_Comparison_Interactive.html
│   └── Model_Radar_Chart.html
│
├── Models/
│   └── training_results.json          # Exported metrics
│
└── mlruns.db                          # MLflow experiment tracking
```

---

## How to Run

```bash
# Activate environment
cd PythonProject
.venv\Scripts\activate

# Run complete pipeline
python main.py
```

All stages will execute automatically and generate reports and visualizations.

---

## Technologies Used

- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Experiment Tracking:** MLflow
- **Report Generation:** ReportLab, WeasyPrint, PDF

---

## Key Insights

1. **Data Quality:** 61.2% of records removed due to outliers (aggressive cleaning)
2. **Model Selection:** Logistic Regression outperforms complex models, indicating linear relationships
3. **Class Balance:** Stratified splitting maintained class distribution in train/test sets
4. **Overfitting:** Tree-based models overfit significantly; Logistic Regression best generalizes
5. **Feature Engineering:** Standard scaling helped linear models converge better

---

## Recommendations

1. **Production Model:** Deploy Logistic Regression (best F1 score 0.7806)
2. **Next Steps:**
   - Feature engineering to improve signal
   - Hyperparameter tuning for other models
   - Cross-validation for more robust evaluation
   - Addressing class imbalance if present
   - External validation on new data

3. **Data:**
   - Investigate outliers before removal
   - Collect more samples to improve model robustness
   - Consider domain expert feature selection

---

Generated: February 16, 2026
