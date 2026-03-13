import io
import base64
import json
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

def df_to_json_safe(df, max_rows=100):
    """Convert dataframe to JSON-safe dict"""
    preview = df.head(max_rows).copy()
    for col in preview.select_dtypes(include=[np.number]).columns:
        preview[col] = preview[col].round(4)
    return {
        'columns': preview.columns.tolist(),
        'data': preview.fillna('').values.tolist(),
        'total_rows': len(df),
        'shown_rows': min(max_rows, len(df))
    }

def generate_pdf_report(analysis_result):
    """Generate a PDF quality report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Title']
    story.append(Paragraph("Data Quality Analysis Report", title_style))
    story.append(Spacer(1, 20))

    # Quality Score
    score = analysis_result.get('quality_score', 0)
    story.append(Paragraph(f"<b>Overall Quality Score: {score}/100</b>", styles['Heading2']))
    story.append(Spacer(1, 10))

    # Dataset Stats
    stats = analysis_result.get('dataset_stats', {})
    story.append(Paragraph("Dataset Overview", styles['Heading2']))
    stat_data = [
        ['Metric', 'Value'],
        ['Total Rows', str(stats.get('rows', ''))],
        ['Total Columns', str(stats.get('columns', ''))],
        ['Numeric Columns', str(stats.get('numeric_columns', ''))],
        ['Categorical Columns', str(stats.get('categorical_columns', ''))],
        ['Memory (KB)', str(stats.get('memory_usage_kb', ''))]
    ]
    t = Table(stat_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f0f0')])
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Missing Values
    missing = analysis_result.get('missing_values', {})
    story.append(Paragraph("Missing Values", styles['Heading2']))
    story.append(Paragraph(f"Total missing: {missing.get('total_missing', 0)} ({missing.get('total_percentage', 0)}%)", styles['Normal']))
    story.append(Spacer(1, 10))

    # Duplicates
    dups = analysis_result.get('duplicates', {})
    story.append(Paragraph("Duplicates", styles['Heading2']))
    story.append(Paragraph(f"Duplicate rows: {dups.get('count', 0)} ({dups.get('percentage', 0)}%)", styles['Normal']))
    story.append(Spacer(1, 10))

    # Suggestions
    suggestions = analysis_result.get('cleaning_suggestions', [])
    if suggestions:
        story.append(Paragraph("Cleaning Recommendations", styles['Heading2']))
        for s in suggestions:
            story.append(Paragraph(f"• [{s['type'].upper()}] {s['action']}", styles['Normal']))
        story.append(Spacer(1, 10))

    doc.build(story)
    buffer.seek(0)
    return buffer

def get_ai_insights(analysis_result, prediction_result=None):
    """Generate AI-style insights text"""
    insights = []
    score = analysis_result.get('quality_score', 0)
    
    if score >= 80:
        insights.append({"icon": "✅", "title": "High Data Quality", "text": "Your dataset shows excellent quality. Models trained on this data are likely to generalize well."})
    elif score >= 60:
        insights.append({"icon": "⚠️", "title": "Moderate Data Quality", "text": "Dataset has some quality issues. Address missing values and outliers before modeling."})
    else:
        insights.append({"icon": "🚨", "title": "Low Data Quality", "text": "Significant data quality problems detected. Clean the data thoroughly before analysis."})

    missing = analysis_result.get('missing_values', {})
    if missing.get('total_percentage', 0) > 10:
        insights.append({"icon": "🔍", "title": "Missing Data Pattern", "text": f"{missing['total_percentage']}% data is missing. Consider median imputation for numeric and mode for categorical features."})

    imbalance = analysis_result.get('class_imbalance', {})
    if imbalance.get('is_imbalanced'):
        insights.append({"icon": "⚖️", "title": "Class Imbalance Detected", "text": f"Imbalance ratio {imbalance.get('imbalance_ratio')}:1. Use SMOTE oversampling or class_weight='balanced' to improve minority class prediction."})

    corr = analysis_result.get('correlation', {})
    if corr.get('high_correlations'):
        n = len(corr['high_correlations'])
        insights.append({"icon": "🔗", "title": "High Feature Correlations", "text": f"{n} highly correlated feature pair(s) found. Consider removing redundant features to reduce multicollinearity."})

    # Churn-specific insights
    if prediction_result:
        prob = prediction_result.get('probability', 0)
        if prob > 0.7:
            insights.append({"icon": "🚨", "title": "High Churn Risk", "text": "Customer shows strong churn signals. Key drivers: short tenure, high monthly charges, month-to-month contract."})
        elif prob > 0.4:
            insights.append({"icon": "⚡", "title": "Moderate Churn Risk", "text": "Customer has moderate retention risk. Proactive engagement via loyalty programs may help."})
        else:
            insights.append({"icon": "💚", "title": "Low Churn Risk", "text": "Customer shows strong retention signals. Long tenure and contract stability are key positive factors."})

    return insights
