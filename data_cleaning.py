"""
Data cleaning and preprocessing pipeline
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from config import CLEANED_DATA_PATH, PRIMARY_TARGET, EXCLUDE_COLUMNS


class DataCleaner:
    """Handle data cleaning and preprocessing"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.label_encoders = {}
        
    def remove_duplicates(self):
        """Remove duplicate rows"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"[OK] Removed {removed} duplicate rows")
        return self
    
    def handle_missing_values(self, numeric_strategy='median', categorical_strategy='most_frequent'):
        """Handle missing values"""
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        if self.numeric_cols:
            self.df[self.numeric_cols] = numeric_imputer.fit_transform(self.df[self.numeric_cols])
        
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        if self.categorical_cols:
            self.df[self.categorical_cols] = categorical_imputer.fit_transform(self.df[self.categorical_cols])
        
        print(f"[OK] Handled missing values (numeric: {numeric_strategy}, categorical: {categorical_strategy})")
        return self
    
    def encode_categorical(self, exclude_cols=None):
        """Encode categorical variables"""
        if exclude_cols is None:
            exclude_cols = []
        
        cols_to_encode = [col for col in self.categorical_cols if col not in exclude_cols]
        
        for col in cols_to_encode:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        if cols_to_encode:
            print(f"[OK] Encoded {len(cols_to_encode)} categorical columns")
        return self
    
    def remove_outliers_iqr(self, columns=None, multiplier=1.5):
        """Remove outliers using IQR method"""
        if columns is None:
            columns = self.numeric_cols
        
        initial_count = len(self.df)
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_count - len(self.df)
        if removed > 0:
            print(f"[OK] Removed {removed} outlier rows (IQR method)")
        return self
    
    def remove_columns(self, columns):
        """Remove specified columns"""
        cols_to_drop = [col for col in columns if col in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self.numeric_cols = [col for col in self.numeric_cols if col not in cols_to_drop]
            self.categorical_cols = [col for col in self.categorical_cols if col not in cols_to_drop]
            print(f"[OK] Removed {len(cols_to_drop)} columns")
        return self
    
    def handle_inconsistencies(self):
        """Handle data inconsistencies"""
        for col in self.numeric_cols:
            if col not in ['Age', 'N_Days', 'Prothrombin']:
                if (self.df[col] < 0).any():
                    initial_count = len(self.df)
                    self.df = self.df[self.df[col] >= 0]
                    removed = initial_count - len(self.df)
                    print(f"[OK] Removed {removed} rows with negative values in {col}")
        
        return self
    
    def get_cleaned_data(self):
        """Return cleaned dataframe"""
        return self.df
    
    def save_cleaned_data(self, path=CLEANED_DATA_PATH):
        """Save cleaned data to CSV"""
        self.df.to_csv(path, index=False)
        print(f"[OK] Cleaned data saved: {path}")
        return self


def clean_cirrhosis_data(df):
    """Apply complete cleaning pipeline"""
    print("\nStarting Data Cleaning Pipeline...")
    
    cleaner = DataCleaner(df)
    
    cleaned_df = (cleaner
                  .remove_duplicates()
                  .handle_missing_values(numeric_strategy='median', categorical_strategy='most_frequent')
                  .handle_inconsistencies()
                  .encode_categorical(exclude_cols=[PRIMARY_TARGET])
                  .remove_columns(EXCLUDE_COLUMNS)
                  .remove_outliers_iqr(multiplier=1.5)
                  .save_cleaned_data()
                  .get_cleaned_data())
    
    print(f"[OK] Cleaning complete! Final shape: {cleaned_df.shape}\n")
    
    return cleaned_df
