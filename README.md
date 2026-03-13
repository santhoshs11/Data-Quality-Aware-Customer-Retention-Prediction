# RetainIQ — Data Quality-Aware Customer Retention Prediction System

A full-stack ML application that combines **data quality analysis** with **customer churn prediction** — built with Flask, scikit-learn, and Chart.js.

---

## 🚀 Features

| Module | Capabilities |
|---|---|
| **Data Quality Engine** | Missing values, duplicates, outliers (IQR/Z-Score), type inconsistencies, class imbalance, correlation heatmap |
| **Quality Score** | 0–100 composite score with grade and recommendations |
| **ML Pipeline** | Logistic Regression, Random Forest, Gradient Boosting — auto-selects best |
| **Prediction API** | Real-time churn probability with risk level and AI insights |
| **PDF Report** | Downloadable quality report |
| **Interactive UI** | Dark cyberpunk dashboard with Chart.js visualizations |

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── app.py              # Flask application & API routes
│   ├── data_quality.py     # Data quality analysis module
│   ├── churn_model.py      # ML model training & prediction
│   └── utils.py            # PDF report, AI insights utilities
├── frontend/
│   ├── templates/
│   │   ├── index.html      # Landing page
│   │   ├── upload.html     # Dataset upload & preview
│   │   ├── dashboard.html  # Data quality dashboard
│   │   ├── prediction.html # Churn prediction form
│   │   └── performance.html# Model performance metrics
│   └── static/
│       ├── css/style.css   # Dark cyberpunk stylesheet
│       └── js/script.js    # Shared JS utilities
├── data/
│   └── telecom_churn.csv   # Sample telecom dataset
├── models/
│   └── churn_model.pkl     # Trained model (auto-generated)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone or extract the project

```bash
cd project
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate sample dataset

```bash
python data/generate_data.py
```

---

## ▶️ Running the Application

```bash
cd backend
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-dataset` | Upload a CSV file |
| `POST` | `/load-default` | Load sample telecom dataset |
| `POST` | `/analyze-data` | Run full data quality analysis |
| `POST` | `/train-model` | Train all ML models |
| `GET`  | `/model-metrics` | Get model evaluation metrics |
| `POST` | `/predict-churn` | Predict churn for one customer |
| `GET`  | `/download-report` | Download PDF quality report |
| `GET`  | `/dataset-preview` | Get dataset preview |
| `GET`  | `/health` | Health check |

---

## 📊 Sample Dataset Format

The telecom churn dataset includes these columns:

```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges, Churn
```

**Churn column**: `Yes` / `No`

---

## 🔌 Prediction API Example

```bash
curl -X POST http://localhost:5000/predict-churn \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5,
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "OnlineSecurity": "No",
    "TechSupport": "No"
  }'
```

**Response:**
```json
{
  "churn": true,
  "probability": 0.7832,
  "confidence": 0.7832,
  "risk_level": "High",
  "model_used": "Random Forest",
  "insights": [...]
}
```

---

## 🎨 Dashboard Pages

1. **Landing** (`/`) — Hero section, feature cards, ML pipeline overview
2. **Upload** (`/upload`) — CSV upload, dataset preview, column stats, model training
3. **Quality** (`/dashboard`) — Quality score gauge, missing/outlier charts, correlation heatmap, AI insights
4. **Predict** (`/prediction`) — Customer attribute form, churn probability, risk gauge, AI insights
5. **Models** (`/performance`) — Confusion matrix, ROC curves, feature importance, model comparison

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask 2.3, Flask-CORS |
| ML | scikit-learn, Gradient Boosting, Random Forest, Logistic Regression |
| Data | pandas, numpy, scipy |
| PDF | ReportLab |
| Frontend | HTML5, Bootstrap 5, Chart.js 4 |
| Fonts | Syne, Space Mono, Inter |

---

## 📈 Model Performance (Sample)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~79% | ~72% | ~85% |
| Random Forest | ~85% | ~80% | ~91% |
| Gradient Boosting | ~84% | ~79% | ~90% |

*Results vary with dataset. Best model is auto-selected.*

---

## 👨‍💻 Built For

Final year Data Science / ML Engineering project demonstrating:
- End-to-end ML pipeline engineering
- Data quality assessment methodology
- REST API design with Flask
- Modern frontend data visualization
- Production-ready model serialization

---

**RetainIQ** — Analyze. Predict. Retain. 🚀
