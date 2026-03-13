import os
import sys
import json
import io
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_quality import DataQualityAnalyzer
from churn_model import ChurnModelTrainer
from utils import df_to_json_safe, generate_pdf_report, get_ai_insights

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'templates'),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend', 'static')
)
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'churn_model.pkl')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

trainer = ChurnModelTrainer()
current_df = None
last_analysis = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_if_exists():
    global trainer
    if os.path.exists(MODEL_PATH):
        try:
            trainer.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load model: {e}")

# ─── Pages ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/performance')
def performance_page():
    return render_template('performance.html')

# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    global current_df, last_analysis
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        current_df = pd.read_csv(filepath)
        preview = df_to_json_safe(current_df)
        return jsonify({
            'success': True,
            'filename': filename,
            'preview': preview,
            'message': f'Dataset uploaded: {len(current_df)} rows, {len(current_df.columns)} columns'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load-default', methods=['POST'])
def load_default():
    global current_df
    default_path = os.path.join(app.config['UPLOAD_FOLDER'], 'telecom_churn.csv')
    if not os.path.exists(default_path):
        return jsonify({'error': 'Default dataset not found'}), 404
    try:
        current_df = pd.read_csv(default_path)
        preview = df_to_json_safe(current_df)
        return jsonify({
            'success': True,
            'filename': 'telecom_churn.csv',
            'preview': preview,
            'message': f'Default dataset loaded: {len(current_df)} rows, {len(current_df.columns)} columns'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-data', methods=['POST'])
def analyze_data():
    global current_df, last_analysis
    df = current_df
    if df is None:
        # Try loading default
        default_path = os.path.join(app.config['UPLOAD_FOLDER'], 'telecom_churn.csv')
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            return jsonify({'error': 'No dataset loaded'}), 400
    try:
        analyzer = DataQualityAnalyzer(df)
        result = analyzer.full_analysis()
        insights = get_ai_insights(result)
        result['ai_insights'] = insights
        # Make JSON serializable
        result = json.loads(json.dumps(result, default=str))
        last_analysis = result
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    global trainer, current_df
    df = current_df
    if df is None:
        default_path = os.path.join(app.config['UPLOAD_FOLDER'], 'telecom_churn.csv')
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            return jsonify({'error': 'No dataset loaded'}), 400
    try:
        trainer = ChurnModelTrainer()
        metrics, best_model = trainer.train(df)
        trainer.save(MODEL_PATH)
        return jsonify({
            'success': True,
            'best_model': best_model,
            'metrics': metrics,
            'message': f'Model trained successfully. Best model: {best_model}'
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/model-metrics', methods=['GET'])
def model_metrics():
    if not trainer.metrics:
        return jsonify({'error': 'No model trained yet. Please train the model first.'}), 404
    return jsonify({
        'metrics': trainer.metrics,
        'best_model': trainer.best_model_name,
        'feature_names': trainer.feature_names
    })

@app.route('/predict-churn', methods=['POST'])
def predict_churn():
    if trainer.best_model is None:
        return jsonify({'error': 'Model not trained. Please train the model first.'}), 400
    try:
        data = request.get_json()
        result = trainer.predict(data)
        # Add AI insights for prediction
        if last_analysis:
            insights = get_ai_insights(last_analysis, result)
        else:
            insights = get_ai_insights({}, result)
        result['insights'] = insights
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/download-report', methods=['GET'])
def download_report():
    if last_analysis is None:
        return jsonify({'error': 'No analysis performed yet'}), 400
    try:
        pdf_buffer = generate_pdf_report(last_analysis)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='data_quality_report.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dataset-preview', methods=['GET'])
def dataset_preview():
    global current_df
    if current_df is None:
        default_path = os.path.join(app.config['UPLOAD_FOLDER'], 'telecom_churn.csv')
        if os.path.exists(default_path):
            current_df = pd.read_csv(default_path)
    if current_df is None:
        return jsonify({'error': 'No dataset'}), 404
    return jsonify(df_to_json_safe(current_df))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': trainer.best_model is not None})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    load_model_if_exists()
    app.run(debug=True, host='0.0.0.0', port=5000)
