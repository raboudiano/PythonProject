# How to View Results

## PDF Reports

### 1. Enhanced EDA Report
**File:** `Reports/Cirrhosis_EDA_Report_Enhanced.pdf`

Contains:
- Dataset overview with key statistics
- Missing data analysis (count and percentage)
- Data type distribution
- Numeric feature distributions (histograms)
- Box plots for outlier detection
- Categorical feature value counts
- Correlation heatmap
- Descriptive statistics table

**Best viewed with:** Adobe Reader, Preview (Mac), or any PDF viewer

---

### 2. Model Comparison Report
**File:** `Reports/Model_Comparison.pdf`

Contains:
- Model accuracy comparison (horizontal bar charts)
- Complete performance summary:
  - Accuracy, Precision, Recall, F1 Score for all models
  - Train vs Test accuracy comparison
- ROC curves for binary classification performance
- Confusion matrices for each model
- Feature importance plots from tree-based models

**Pages:**
1. Model accuracy and F1 score comparison
2. 4-panel training results summary
3. ROC curves
4-9. Individual confusion matrices and feature importance

---

## Interactive Visualizations (HTML)

### 3. Model Comparison Interactive Chart
**File:** `Reports/Model_Comparison_Interactive.html`

**How to use:**
- Open in any web browser
- Hover over bars to see exact values
- Click legend items to show/hide models
- Use browser's export/print features to save

**Shows:**
- 4 key metrics side-by-side for each model
- Interactive tooltips with detailed values
- Color-coded bars for easy distinction

---

### 4. Model Performance Radar Chart
**File:** `Reports/Model_Radar_Chart.html`

**How to use:**
- Open in any web browser
- Drag to rotate the chart
- Hover to highlight individual models
- Zoom using mouse wheel
- Click legend to show/hide models

**Shows:**
- Multi-dimensional view of performance
- 4 metrics displayed as axes:
  - Test Accuracy
  - Precision
  - Recall
  - F1 Score
- All 6 models on same chart for comparison
- Area fill shows relative performance

---

## CSV Data Files

### Original Dataset
**File:** `Data/cirrhosis.csv`
- 418 rows, 20 columns
- Raw data with missing values
- Target variable: Status (C, CL, D)

### Cleaned Dataset
**File:** `Data/cirrhosis_cleaned.csv`
- 163 rows, 19 columns
- All missing values imputed
- All categorical variables encoded
- Outliers removed
- Ready for model training

---

## JSON Results

**File:** `Models/training_results.json`

Contains:
- All model metrics for each trained model
- Fields: train_accuracy, test_accuracy, test_precision, test_recall, test_f1
- Example:
```json
{
  "Logistic Regression": {
    "metrics": {
      "train_accuracy": 0.8846,
      "test_accuracy": 0.7879,
      "test_precision": 0.7779,
      "test_recall": 0.7879,
      "test_f1": 0.7806
    }
  }
}
```

---

## MLflow Tracking

**Database:** `mlruns.db`

To view MLflow UI:
```bash
mlflow ui
```

Then navigate to http://localhost:5000 in your browser

Shows:
- Each model training run
- Hyperparameters used
- Metrics logged during training
- Experiment history

---

## Viewing Results in VS Code

1. **PDFs:** Use VS Code's Simple Browser or external PDF viewer
2. **HTML:** Right-click file → "Open with Preview" or "Open in Default Browser"
3. **CSV:** Use "Data Wrangler" extension or "Excel Viewer" for better visualization
4. **JSON:** Built-in JSON viewer with syntax highlighting

---

## Recommended Viewing Order

1. Start with `PIPELINE_SUMMARY.md` for overview
2. Open `Reports/Cirrhosis_EDA_Report_Enhanced.pdf` for data insights
3. View `Reports/Model_Comparison.pdf` for model performance details
4. Explore `Reports/Model_Comparison_Interactive.html` for interactive comparison
5. Check `Reports/Model_Radar_Chart.html` for multi-dimensional view
6. Review cleaned data in `Data/cirrhosis_cleaned.csv`
7. Check `Models/training_results.json` for programmatic access to metrics

---

## Model Selection Guide

Based on the results:

**Choose Logistic Regression if:**
- You need the best test accuracy (0.7879)
- You want the best F1 score (0.7806)
- You need model interpretability
- You want fastest inference
- You need to avoid overfitting

**Choose tree-based models if:**
- You need feature interactions
- You have more data to prevent overfitting
- You need non-linear decision boundaries

---

## Troubleshooting

**Can't open PDF:** 
- Install Adobe Reader or use built-in Preview

**HTML won't display:**
- Open with Chrome, Firefox, or Edge
- Right-click → "Open with Browser"

**CSV won't load:**
- Use Excel or Data Wrangler extension in VS Code

**MLflow UI won't start:**
- Ensure MLflow is installed: `pip install mlflow`
- Check port 5000 is not in use
- Run: `mlflow ui --host localhost --port 5000`

---

Happy analyzing!
