"""
Data loading and exploration utilities
"""
import pandas as pd
import numpy as np
from config import DATA_PATH, CLEANED_DATA_PATH


def load_data():
    """Load the cirrhosis dataset"""
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def get_basic_info(df):
    """Get basic information about the dataset"""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage": df.memory_usage(deep=True).sum() / 1024**2  # MB
    }


def get_descriptive_stats(df):
    """Get descriptive statistics"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    return {
        "numeric": df[numeric_cols].describe().to_dict(),
        "categorical": {col: df[col].value_counts().to_dict() for col in categorical_cols}
    }


def get_missing_data_analysis(df):
    """Analyze missing data"""
    missing_df = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    return missing_df


def get_correlation_matrix(df):
    """Get correlation matrix for numeric columns"""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def get_distribution_info(df):
    """Get distribution information for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dist_info = {}
    
    for col in numeric_cols:
        dist_info[col] = {
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std()
        }
    
    return dist_info
