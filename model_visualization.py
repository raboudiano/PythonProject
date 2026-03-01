"""
Model visualization utilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages


def plot_model_comparison(comparison_df, figsize=(12, 6)):
    """Plot model comparison metrics"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].barh(comparison_df['Model'], comparison_df['Test Accuracy'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Test Accuracy Comparison')
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(comparison_df['Model'], comparison_df['F1 Score'], color='coral', alpha=0.7)
    axes[1].set_xlabel('F1 Score')
    axes[1].set_title('F1 Score Comparison')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_metrics_radar(comparison_df):
    """Create radar chart for model comparison"""
    metrics = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig = go.Figure()
    
    for idx, row in comparison_df.iterrows():
        values = [row[metric] for metric in metrics] + [row[metrics[0]]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title='Model Performance Radar Chart',
        height=700,
        showlegend=True
    )
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, figsize=(8, 6)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return fig


def plot_roc_curves(results_dict, figsize=(10, 7)):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, result in results_dict.items():
        if result.get('y_pred_proba') is not None:
            y_test = result.get('y_test')
            if y_test is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
                        roc_auc = result['metrics'].get('test_roc_auc', auc(fpr, tpr))
                        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
                except:
                    pass
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    return fig


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 6)):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} doesn't have feature_importances_")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importances[indices], color='green', alpha=0.7)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance - {type(model).__name__}')
    ax.grid(axis='x', alpha=0.3)
    
    return fig


def create_interactive_comparison(comparison_df):
    """Create interactive comparison plot"""
    fig = px.bar(comparison_df, x='Model', y=['Test Accuracy', 'Precision', 'Recall', 'F1 Score'],
                 barmode='group', title='Model Performance Comparison',
                 labels={'value': 'Score', 'variable': 'Metric'},
                 height=500)
    
    fig.update_layout(hovermode='x unified')
    return fig


def plot_training_comparison_summary(comparison_df, figsize=(14, 8)):
    """Create comprehensive comparison summary"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Training Results Summary', fontsize=16, fontweight='bold')
    
    axes[0, 0].barh(comparison_df['Model'], comparison_df['Test Accuracy'], color='#1f77b4', alpha=0.7)
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_title('Test Accuracy')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    axes[0, 1].barh(comparison_df['Model'], comparison_df['Precision'], color='#ff7f0e', alpha=0.7)
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_title('Precision')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    axes[1, 0].barh(comparison_df['Model'], comparison_df['Recall'], color='#2ca02c', alpha=0.7)
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_title('Recall')
    axes[1, 0].set_xlim([0, 1])
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    axes[1, 1].barh(comparison_df['Model'], comparison_df['F1 Score'], color='#d62728', alpha=0.7)
    axes[1, 1].set_xlabel('Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_visualizations_to_pdf(comparison_df, results_dict, output_path='Reports/Model_Comparison.pdf'):
    """Save all visualizations to PDF"""
    
    with PdfPages(output_path) as pdf:
        fig = plot_model_comparison(comparison_df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        fig = plot_training_comparison_summary(comparison_df)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        fig = plot_roc_curves(results_dict)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        for model_name, result in results_dict.items():
            if result.get('y_test') is not None:
                fig = plot_confusion_matrix(result['y_test'], result['y_pred_test'], model_name)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        for model_name, result in results_dict.items():
            model = result['model']
            if hasattr(model, 'feature_importances_'):
                try:
                    feature_names = [f'Feature_{i}' for i in range(model.n_features_in_)]
                except:
                    continue
                
                fig = plot_feature_importance(model, feature_names)
                if fig:
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
    
    print(f"[OK] Visualizations saved: {output_path}")
