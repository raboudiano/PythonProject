"""
Main execution script - Complete pipeline
"""
import os
import sys
from pathlib import Path

# Create necessary directories
os.makedirs('Reports', exist_ok=True)
os.makedirs('Models', exist_ok=True)

# Import modules
from data_utils import load_data, get_basic_info, get_missing_data_analysis, get_correlation_matrix
from eda_visualization import create_eda_pdf
from data_cleaning import clean_cirrhosis_data
from model_training import train_models
from model_visualization import (plot_model_comparison, plot_training_comparison_summary,
                                plot_roc_curves, save_visualizations_to_pdf)
import pandas as pd


def main():
    """Execute complete pipeline"""
    
    print("=" * 70)
    print("CIRRHOSIS PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 70)
    
    # ==================== PHASE 1: EDA ====================
    print("\n" + "="*70)
    print("PHASE 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Basic info
    info = get_basic_info(df)
    print(f"\nDataset Shape: {info['shape']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    
    # Missing data analysis
    print("\nMissing Data Analysis:")
    missing_df = get_missing_data_analysis(df)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("[OK] No missing values found")
    
    # Create comprehensive EDA report
    print("\nCreating EDA Report (PDF)...")
    create_eda_pdf(df, 'Reports/Cirrhosis_EDA_Report_Enhanced.pdf')
    
    # Correlation analysis
    print("\nCorrelation Matrix Analysis:")
    corr_matrix = get_correlation_matrix(df)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print("\nHigh correlation pairs (|r| > 0.7):")
        for pair in high_corr_pairs[:5]:
            print(f"  {pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.3f}")
    
    # ==================== PHASE 2: DATA CLEANING ====================
    print("\n" + "="*70)
    print("PHASE 2: DATA CLEANING & PREPROCESSING")
    print("="*70)
    
    cleaned_df = clean_cirrhosis_data(df)
    
    print(f"\nCleaning Summary:")
    print(f"  Original shape: {df.shape}")
    print(f"  Cleaned shape: {cleaned_df.shape}")
    print(f"  Rows removed: {df.shape[0] - cleaned_df.shape[0]}")
    
    # ==================== PHASE 3: MODEL TRAINING ====================
    print("\n" + "="*70)
    print("PHASE 3: MODEL TRAINING WITH MLflow")
    print("="*70)
    
    trainer = train_models(cleaned_df)
    
    # ==================== PHASE 4: MODEL COMPARISON ====================
    print("\n" + "="*70)
    print("PHASE 4: MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    comparison_df = trainer.get_comparison_df()
    print("\nModel Performance Comparison:\n")
    print(comparison_df.to_string(index=False))
    
    # Best model
    best_model_name, best_result = trainer.get_best_model()
    print(f"\n[BEST] Model: {best_model_name}")
    print(f"   F1 Score: {best_result['metrics']['test_f1']:.4f}")
    print(f"   Accuracy: {best_result['metrics']['test_accuracy']:.4f}")
    
    # ==================== PHASE 5: VISUALIZATIONS ====================
    print("\n" + "="*70)
    print("PHASE 5: GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Save all visualizations to PDF
    save_visualizations_to_pdf(comparison_df, trainer.results)
    
    # Export results
    trainer.export_results()
    
    # ==================== PHASE 6: INTERACTIVE VISUALIZATIONS ====================
    print("\n" + "="*70)
    print("PHASE 6: INTERACTIVE VISUALIZATIONS")
    print("="*70)
    
    from model_visualization import create_interactive_comparison
    
    interactive_fig = create_interactive_comparison(comparison_df)
    interactive_fig.write_html('Reports/Model_Comparison_Interactive.html')
    print("[OK] Interactive comparison saved: Reports/Model_Comparison_Interactive.html")
    
    # Radar chart
    from model_visualization import plot_model_metrics_radar
    radar_fig = plot_model_metrics_radar(comparison_df)
    radar_fig.write_html('Reports/Model_Radar_Chart.html')
    print("[OK] Radar chart saved: Reports/Model_Radar_Chart.html")
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    
    print("\nGenerated Files:")
    print("  Reports:")
    print("    - Reports/Cirrhosis_EDA_Report_Enhanced.pdf")
    print("    - Reports/Model_Comparison.pdf")
    print("    - Reports/Model_Comparison_Interactive.html")
    print("    - Reports/Model_Radar_Chart.html")
    print("  Data:")
    print("    - Data/cirrhosis_cleaned.csv")
    print("  Models:")
    print("    - Models/training_results.json")
    print("    - MLflow tracking in mlruns.db")
    
    print("\nKey Metrics:")
    print(f"  Best Model: {best_model_name}")
    print(f"  Test Accuracy: {best_result['metrics']['test_accuracy']:.4f}")
    print(f"  F1 Score: {best_result['metrics']['test_f1']:.4f}")
    print(f"  Precision: {best_result['metrics']['test_precision']:.4f}")
    print(f"  Recall: {best_result['metrics']['test_recall']:.4f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
