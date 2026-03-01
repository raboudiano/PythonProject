"""
MLflow model training and comparison
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from config import PRIMARY_TARGET, MODEL_RANDOM_STATE, TEST_SIZE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


class ModelTrainer:
    """Handle model training and MLflow tracking"""
    
    def __init__(self, df, target_col=PRIMARY_TARGET):
        self.df = df.copy()
        self.target_col = target_col
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    def prepare_data(self, test_size=TEST_SIZE):
        """Prepare data for training"""
        le = LabelEncoder()
        y = le.fit_transform(self.df[self.target_col])
        
        X = self.df.drop(columns=[self.target_col])
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=MODEL_RANDOM_STATE, stratify=y
        )
        
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        print(f"[OK] Data prepared: Train size={len(self.X_train)}, Test size={len(self.X_test)}")
        return self
    
    def train_model(self, model_name, model, hyperparams=None):
        """Train a single model with MLflow tracking"""
        
        with mlflow.start_run(run_name=model_name):
            # Explicitly set model name as a tag
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("mlflow.runName", model_name)
            
            if hyperparams:
                mlflow.log_params(hyperparams)
            
            model.fit(self.X_train, self.y_train)
            
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if model.predict_proba(self.X_test).shape[1] > 1 else None
                try:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
                except:
                    roc_auc = None
            else:
                y_pred_proba = None
                roc_auc = None
            
            metrics = {
                'train_accuracy': accuracy_score(self.y_train, y_pred_train),
                'test_accuracy': accuracy_score(self.y_test, y_pred_test),
                'test_precision': precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
                'test_recall': recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
                'test_f1': f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            }
            
            if roc_auc is not None:
                metrics['test_roc_auc'] = roc_auc
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Skip log_model to avoid subprocess issues
            # mlflow.sklearn.log_model(model, f"model_{model_name}")
            
            self.models[model_name] = model
            self.results[model_name] = {
                'model': model,
                'metrics': metrics,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'y_test': self.y_test,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"[OK] {model_name} - Accuracy: {metrics['test_accuracy']:.4f}, F1: {metrics['test_f1']:.4f}")
        
        return self
    
    def train_all_models(self):
        """Train all models"""
        print("\nTraining Models with MLflow...\n")
        
        models_config = {
            'Logistic Regression': (
                LogisticRegression(random_state=MODEL_RANDOM_STATE, max_iter=1000),
                {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}
            ),
            'Random Forest': (
                RandomForestClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE, n_jobs=-1),
                {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2}
            ),
            'Gradient Boosting': (
                GradientBoostingClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE),
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            ),
            'XGBoost': (
                XGBClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE, verbosity=0),
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
            ),
            'LightGBM': (
                LGBMClassifier(n_estimators=100, random_state=MODEL_RANDOM_STATE, verbose=-1),
                {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5}
            ),
            'SVM': (
                SVC(kernel='rbf', random_state=MODEL_RANDOM_STATE, probability=True),
                {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
            )
        }
        
        for model_name, (model, hyperparams) in models_config.items():
            self.train_model(model_name, model, hyperparams)
        
        return self
    
    def get_best_model(self):
        """Get best model based on test F1 score"""
        best_model_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]['metrics'].get('test_f1', 0)
        )
        return best_model_name, self.results[best_model_name]
    
    def get_comparison_df(self):
        """Get comparison dataframe of all models"""
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Train Accuracy': result['metrics']['train_accuracy'],
                'Test Accuracy': result['metrics']['test_accuracy'],
                'Precision': result['metrics']['test_precision'],
                'Recall': result['metrics']['test_recall'],
                'F1 Score': result['metrics']['test_f1'],
                'ROC-AUC': result['metrics'].get('test_roc_auc', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('F1 Score', ascending=False)
        return comparison_df
    
    def export_results(self, output_path='Models/training_results.json'):
        """Export results to JSON"""
        export_data = {}
        
        for model_name, result in self.results.items():
            export_data[model_name] = {
                'metrics': {k: float(v) if v is not None else None 
                           for k, v in result['metrics'].items()}
            }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"[OK] Results exported: {output_path}")
        return self


def train_models(cleaned_df):
    """Train all models and return trainer object"""
    trainer = ModelTrainer(cleaned_df)
    trainer.prepare_data()
    trainer.train_all_models()
    
    return trainer
