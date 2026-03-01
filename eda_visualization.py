"""
EDA visualization utilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def create_eda_pdf(df, output_path, dpi=300):
    """Create comprehensive EDA report as PDF"""
    
    with PdfPages(output_path) as pdf:
        # Page 1: Title & Basic Info
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Cirrhosis Patient Survival Prediction - EDA Report', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        info_text = f"""
        Dataset Overview
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        Total Records: {df.shape[0]:,}
        Total Features: {df.shape[1]}
        Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        
        Missing Values: {df.isnull().sum().sum()}
        Duplicate Rows: {df.duplicated().sum()}
        
        Numeric Features: {len(df.select_dtypes(include=[np.number]).columns)}
        Categorical Features: {len(df.select_dtypes(include=['object']).columns)}
        
        Data Types:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        
        for dtype in df.dtypes.unique():
            count = len(df.select_dtypes(include=[dtype]).columns)
            info_text += f"        {str(dtype)}: {count}\n"
        
        ax.text(0.1, 0.95, info_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Missing Data Analysis
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Missing Data Analysis', fontsize=16, fontweight='bold')
        
        # Missing data bar plot
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        if len(missing_data) > 0:
            axes[0, 0].barh(range(len(missing_data)), missing_data.values, color='coral')
            axes[0, 0].set_yticks(range(len(missing_data)))
            axes[0, 0].set_yticklabels(missing_data.index)
            axes[0, 0].set_xlabel('Count')
            axes[0, 0].set_title('Missing Values Count')
            axes[0, 0].grid(axis='x', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', 
                          ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Missing Values Count')
        
        # Missing percentage
        missing_percent = (df.isnull().sum() / len(df)) * 100
        missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
        
        if len(missing_percent) > 0:
            axes[0, 1].barh(range(len(missing_percent)), missing_percent.values, color='skyblue')
            axes[0, 1].set_yticks(range(len(missing_percent)))
            axes[0, 1].set_yticklabels(missing_percent.index)
            axes[0, 1].set_xlabel('Percentage (%)')
            axes[0, 1].set_title('Missing Values Percentage')
            axes[0, 1].grid(axis='x', alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', 
                          ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Missing Values Percentage')
        
        # Data types distribution
        dtype_counts = df.dtypes.value_counts()
        axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('Data Types Distribution')
        
        # Dataset shape info
        axes[1, 1].axis('off')
        shape_text = f"Shape: {df.shape}\nRecords × Features\n\n" + \
                    f"Size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n" + \
                    f"Duplicates: {df.duplicated().sum()}"
        axes[1, 1].text(0.5, 0.5, shape_text, ha='center', va='center', 
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Numeric Features - Distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for i in range(0, len(numeric_cols), 6):
            fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
            fig.suptitle('Numeric Features - Distributions (Part {})'.format(i//6 + 1), 
                        fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols[i:i+6]):
                if idx < len(axes):
                    axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                    axes[idx].set_title(col)
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Frequency')
                    axes[idx].grid(alpha=0.3)
            
            # Hide empty subplots
            for idx in range(len(numeric_cols[i:i+6]), len(axes)):
                axes[idx].axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Page: Box plots for outlier detection
        for i in range(0, len(numeric_cols), 6):
            fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
            fig.suptitle('Numeric Features - Box Plots (Part {})'.format(i//6 + 1), 
                        fontsize=14, fontweight='bold')
            axes = axes.flatten()
            
            for idx, col in enumerate(numeric_cols[i:i+6]):
                if idx < len(axes):
                    axes[idx].boxplot(df[col].dropna())
                    axes[idx].set_title(col)
                    axes[idx].set_ylabel('Value')
                    axes[idx].grid(alpha=0.3)
            
            for idx in range(len(numeric_cols[i:i+6]), len(axes)):
                axes[idx].axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Categorical Features
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            for i in range(0, len(categorical_cols), 6):
                fig, axes = plt.subplots(3, 2, figsize=(11, 8.5))
                fig.suptitle('Categorical Features - Value Counts (Part {})'.format(i//6 + 1), 
                            fontsize=14, fontweight='bold')
                axes = axes.flatten()
                
                for idx, col in enumerate(categorical_cols[i:i+6]):
                    if idx < len(axes):
                        value_counts = df[col].value_counts()
                        axes[idx].barh(range(len(value_counts)), value_counts.values, color='lightseagreen')
                        axes[idx].set_yticks(range(len(value_counts)))
                        axes[idx].set_yticklabels(value_counts.index)
                        axes[idx].set_title(col)
                        axes[idx].set_xlabel('Count')
                        axes[idx].grid(axis='x', alpha=0.3)
                
                for idx in range(len(categorical_cols[i:i+6]), len(axes)):
                    axes[idx].axis('off')
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(11, 8.5))
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Correlation Matrix - Numeric Features', fontsize=14, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Descriptive Statistics
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Descriptive Statistics - Numeric Features', fontsize=14, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        stats_df = df.select_dtypes(include=[np.number]).describe().T
        stats_df = stats_df.round(3)
        
        table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                        rowLabels=stats_df.index, cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] EDA Report generated: {output_path}")


def plot_target_distribution(df, target_col, figsize=(10, 5)):
    """Plot target variable distribution"""
    fig, ax = plt.subplots(figsize=figsize)
    
    if df[target_col].dtype == 'object':
        df[target_col].value_counts().plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
    else:
        ax.hist(df[target_col], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.3)
    
    return fig


def plot_feature_target_relationship(df, feature_cols, target_col, figsize=(14, 10)):
    """Plot relationship between features and target"""
    numeric_cols = [col for col in feature_cols if df[col].dtype in [np.int64, np.float64]]
    n_plots = min(len(numeric_cols), 12)
    
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:n_plots]):
        axes[idx].scatter(df[col], df[target_col], alpha=0.5, s=30)
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(target_col)
        axes[idx].grid(alpha=0.3)
    
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'Feature-Target Relationships', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
