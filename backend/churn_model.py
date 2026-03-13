import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import joblib
import os

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.metrics = {}

    def preprocess(self, df):
        data = df.copy()
        # Drop ID columns
        for drop_col in ['customerID', 'CustomerID', 'customer_id']:
            if drop_col in data.columns:
                data = data.drop(drop_col, axis=1)
        # Drop duplicates
        data = data.drop_duplicates().reset_index(drop=True)
        # Fix TotalCharges type if needed
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Encode target
        if 'Churn' in data.columns:
            le = LabelEncoder()
            data['Churn'] = le.fit_transform(data['Churn'].fillna('No').astype(str))
            self.encoders['Churn'] = le
        # Encode all non-numeric columns
        for col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                le = LabelEncoder()
                data[col] = data[col].fillna('Unknown').astype(str)
                data[col] = le.fit_transform(data[col])
                self.encoders[col] = le
        # Fill remaining NaN with median
        for col in data.columns:
            if data[col].isna().any():
                med = data[col].median()
                data[col] = data[col].fillna(med if not np.isnan(med) else 0)
        return data

    def train(self, df):
        data = self.preprocess(df)
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        model_defs = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        best_auc = 0
        for name, model in model_defs.items():
            if name == 'Logistic Regression':
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
                y_prob = model.predict_proba(X_test_sc)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'precision': round(float(precision_score(y_test, y_pred)), 4),
                'recall': round(float(recall_score(y_test, y_pred)), 4),
                'f1': round(float(f1_score(y_test, y_pred)), 4),
                'roc_auc': round(float(roc_auc_score(y_test, y_prob)), 4),
                'confusion_matrix': cm.tolist(),
                'roc_curve': {
                    'fpr': [round(x, 4) for x in fpr.tolist()],
                    'tpr': [round(x, 4) for x in tpr.tolist()]
                }
            }

            if hasattr(model, 'feature_importances_'):
                fi = dict(zip(self.feature_names, model.feature_importances_.tolist()))
                metrics['feature_importance'] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10])
            elif hasattr(model, 'coef_'):
                fi = dict(zip(self.feature_names, np.abs(model.coef_[0]).tolist()))
                metrics['feature_importance'] = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10])

            self.models[name] = model
            self.metrics[name] = metrics

            if metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                self.best_model_name = name
                self.best_model = model

        return self.metrics, self.best_model_name

    def predict(self, input_data):
        if self.best_model is None:
            raise ValueError("Model not trained yet")
        df = pd.DataFrame([input_data])
        for col in df.columns:
            if col in self.encoders:
                le = self.encoders[col]
                val = str(df[col].iloc[0])
                df[col] = le.transform([val])[0] if val in le.classes_ else 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names]
        # Ensure numeric
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        if self.best_model_name == 'Logistic Regression':
            df_sc = self.scaler.transform(df)
            prob = self.best_model.predict_proba(df_sc)[0][1]
            pred = self.best_model.predict(df_sc)[0]
        else:
            prob = self.best_model.predict_proba(df)[0][1]
            pred = self.best_model.predict(df)[0]

        return {
            'churn': bool(pred == 1),
            'probability': round(float(prob), 4),
            'confidence': round(float(max(prob, 1 - prob)), 4),
            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low',
            'model_used': self.best_model_name
        }

    def save(self, path='models/churn_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'models': self.models, 'best_model': self.best_model,
            'best_model_name': self.best_model_name, 'scaler': self.scaler,
            'encoders': self.encoders, 'feature_names': self.feature_names,
            'metrics': self.metrics
        }, path)

    def load(self, path='models/churn_model.pkl'):
        data = joblib.load(path)
        self.models = data['models']; self.best_model = data['best_model']
        self.best_model_name = data['best_model_name']; self.scaler = data['scaler']
        self.encoders = data['encoders']; self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        return self
