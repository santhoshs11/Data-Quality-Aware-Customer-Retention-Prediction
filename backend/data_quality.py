import pandas as pd
import numpy as np
from scipy import stats
import json
import io
import base64

class DataQualityAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self.report = {}
        
    def analyze_missing_values(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        result = {}
        for col in self.df.columns:
            result[col] = {
                'count': int(missing[col]),
                'percentage': float(missing_pct[col])
            }
        total_missing = missing.sum()
        total_pct = (total_missing / (len(self.df) * len(self.df.columns)) * 100)
        return {
            'by_column': result,
            'total_missing': int(total_missing),
            'total_percentage': round(float(total_pct), 2),
            'columns_with_missing': [c for c in self.df.columns if missing[c] > 0]
        }

    def analyze_duplicates(self):
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round(dup_count / len(self.df) * 100, 2)
        return {
            'count': dup_count,
            'percentage': dup_pct
        }

    def analyze_outliers(self, method='iqr'):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}
        total_outliers = 0
        for col in numeric_cols:
            series = self.df[col].dropna()
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = series[(series < lower) | (series > upper)]
            else:
                z = np.abs(stats.zscore(series))
                outliers = series[z > 3]
            count = len(outliers)
            total_outliers += count
            outlier_info[col] = {
                'count': count,
                'percentage': round(count / len(series) * 100, 2)
            }
        return {
            'by_column': outlier_info,
            'total_outliers': total_outliers,
            'method': method
        }

    def analyze_data_types(self):
        type_info = {}
        issues = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            type_info[col] = dtype
            # Check for numeric stored as object
            if dtype == 'object':
                try:
                    self.df[col].dropna().astype(float)
                    issues.append({'column': col, 'issue': 'Numeric data stored as string', 'suggested_type': 'float64'})
                except:
                    pass
        return {'types': type_info, 'issues': issues}

    def analyze_class_imbalance(self, target_col=None):
        if target_col is None:
            # Try to auto-detect target
            for col in ['Churn', 'churn', 'target', 'label']:
                if col in self.df.columns:
                    target_col = col
                    break
        if target_col is None or target_col not in self.df.columns:
            return {'status': 'no_target_found', 'imbalance_ratio': None}
        
        vc = self.df[target_col].value_counts()
        ratio = round(vc.max() / vc.min(), 2) if vc.min() > 0 else None
        imbalanced = ratio > 2 if ratio else False
        return {
            'target_column': target_col,
            'distribution': vc.to_dict(),
            'imbalance_ratio': ratio,
            'is_imbalanced': imbalanced,
            'recommendation': 'Consider SMOTE or class weights' if imbalanced else 'Classes are balanced'
        }

    def analyze_correlation(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {'matrix': {}, 'high_correlations': []}
        corr_matrix = numeric_df.corr().round(3)
        high_corr = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = abs(corr_matrix.iloc[i, j])
                if val > 0.75:
                    high_corr.append({
                        'feature1': cols[i],
                        'feature2': cols[j],
                        'correlation': round(float(corr_matrix.iloc[i, j]), 3)
                    })
        return {
            'matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr,
            'columns': cols
        }

    def compute_quality_score(self, missing_info, dup_info, outlier_info, type_info, imbalance_info):
        score = 100.0
        # Deduct for missing values (up to 30 points)
        missing_penalty = min(30, missing_info['total_percentage'] * 3)
        score -= missing_penalty
        # Deduct for duplicates (up to 20 points)
        dup_penalty = min(20, dup_info['percentage'] * 2)
        score -= dup_penalty
        # Deduct for outliers (up to 20 points)
        numeric_cols = self.df.select_dtypes(include=[np.number]).shape[1]
        if numeric_cols > 0:
            avg_outlier_pct = sum(v['percentage'] for v in outlier_info['by_column'].values()) / max(numeric_cols, 1)
            outlier_penalty = min(20, avg_outlier_pct * 2)
            score -= outlier_penalty
        # Deduct for type issues (up to 15 points)
        type_penalty = min(15, len(type_info['issues']) * 5)
        score -= type_penalty
        # Deduct for class imbalance (up to 15 points)
        if imbalance_info.get('is_imbalanced'):
            ratio = imbalance_info.get('imbalance_ratio', 1)
            imbalance_penalty = min(15, (ratio - 2) * 2)
            score -= imbalance_penalty
        score = max(0, min(100, score))
        return round(score, 1)

    def generate_cleaning_suggestions(self, missing_info, dup_info, outlier_info, type_info):
        suggestions = []
        if missing_info['total_missing'] > 0:
            for col, info in missing_info['by_column'].items():
                if info['percentage'] > 0:
                    if info['percentage'] > 50:
                        suggestions.append({'type': 'critical', 'action': f'Drop column "{col}" — {info["percentage"]}% missing'})
                    elif info['percentage'] > 10:
                        suggestions.append({'type': 'warning', 'action': f'Impute "{col}" with median/mode — {info["percentage"]}% missing'})
                    else:
                        suggestions.append({'type': 'info', 'action': f'Fill "{col}" missing values — {info["percentage"]}% missing'})
        if dup_info['count'] > 0:
            suggestions.append({'type': 'warning', 'action': f'Remove {dup_info["count"]} duplicate rows ({dup_info["percentage"]}% of data)'})
        for col, info in outlier_info['by_column'].items():
            if info['percentage'] > 5:
                suggestions.append({'type': 'warning', 'action': f'Cap or remove outliers in "{col}" — {info["percentage"]}% outliers'})
        for issue in type_info['issues']:
            suggestions.append({'type': 'info', 'action': f'Convert "{issue["column"]}" to {issue["suggested_type"]}'})
        return suggestions

    def full_analysis(self):
        missing = self.analyze_missing_values()
        duplicates = self.analyze_duplicates()
        outliers = self.analyze_outliers()
        dtypes = self.analyze_data_types()
        imbalance = self.analyze_class_imbalance()
        correlation = self.analyze_correlation()
        score = self.compute_quality_score(missing, duplicates, outliers, dtypes, imbalance)
        suggestions = self.generate_cleaning_suggestions(missing, duplicates, outliers, dtypes)
        
        # Dataset stats
        stats_info = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': int(self.df.select_dtypes(include=[np.number]).shape[1]),
            'categorical_columns': int(self.df.select_dtypes(include=['object']).shape[1]),
            'memory_usage_kb': round(self.df.memory_usage(deep=True).sum() / 1024, 2)
        }

        # Column info
        col_info = []
        for col in self.df.columns:
            col_info.append({
                'name': col,
                'dtype': str(self.df[col].dtype),
                'non_null': int(self.df[col].notna().sum()),
                'unique': int(self.df[col].nunique()),
                'sample': str(self.df[col].dropna().iloc[0]) if len(self.df[col].dropna()) > 0 else 'N/A'
            })

        return {
            'quality_score': score,
            'dataset_stats': stats_info,
            'column_info': col_info,
            'missing_values': missing,
            'duplicates': duplicates,
            'outliers': outliers,
            'data_types': dtypes,
            'class_imbalance': imbalance,
            'correlation': correlation,
            'cleaning_suggestions': suggestions
        }
