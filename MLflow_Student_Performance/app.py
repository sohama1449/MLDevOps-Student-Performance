"""
Flask Web Application & REST API for Student Performance Prediction.
Provides endpoints:
- POST /predict: Predicts student Pass/Fail status and confidence score.
- GET /: Interactive web dashboard for model evaluation & live testing.
- GET /health: Service health check.
- POST /detect-drift: On-demand data drift detection & automated retraining trigger.
- GET /drift-report: Returns latest drift detection metrics.
"""

import os
import sys

# Ensure src directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import json
import joblib
from flask import Flask, request, jsonify, render_template_string

from predict import predict_single, get_model_and_preprocessor, MODEL_PATH, METADATA_PATH
from drift_detection import detect_data_drift, run_drift_pipeline_and_retrain, REPORTS_DIR

app = Flask(__name__)

HTML_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Prediction | MLOps MLflow Pipeline</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --card-bg: rgba(22, 30, 49, 0.75);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #4f46e5;
            --accent-glow: #6366f1;
            --pass-color: #10b981;
            --fail-color: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body {
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg) 70%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 40px 20px;
        }
        .container { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 40px; }
        header h1 {
            font-size: 2.6rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a5b4fc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        header p { color: var(--text-muted); font-size: 1.1rem; }
        .badge-grid {
            display: flex; justify-content: center; gap: 12px; margin-top: 15px; flex-wrap: wrap;
        }
        .badge {
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #c7d2fe;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 500;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 28px;
            margin-bottom: 30px;
        }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px;
            backdrop-filter: blur(16px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            font-size: 1.35rem; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;
        }
        .form-group { margin-bottom: 18px; }
        label { display: block; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 6px; }
        input, select {
            width: 100%;
            padding: 12px 16px;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input:focus, select:focus { outline: none; border-color: var(--accent-glow); }
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, var(--accent), var(--accent-glow));
            border: none;
            border-radius: 12px;
            color: #fff;
            font-size: 1.05rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            transition: all 0.2s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6); }
        .btn-secondary {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border);
            box-shadow: none;
            margin-top: 12px;
        }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); transform: translateY(-1px); }
        .result-box {
            text-align: center;
            padding: 30px 20px;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px dashed var(--border);
            margin-top: 20px;
        }
        .pred-pill {
            display: inline-block;
            font-size: 2.2rem;
            font-weight: 700;
            padding: 10px 36px;
            border-radius: 9999px;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
        }
        .pred-Pass { background: rgba(16, 185, 129, 0.2); color: var(--pass-color); border: 2px solid var(--pass-color); }
        .pred-Fail { background: rgba(239, 68, 68, 0.2); color: var(--fail-color); border: 2px solid var(--fail-color); }
        .confidence-meter {
            font-size: 1.15rem; color: var(--text-muted); font-weight: 500;
        }
        .api-tester {
            background: #0f172a;
            padding: 16px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.88rem;
            color: #38bdf8;
            overflow-x: auto;
            margin-top: 15px;
            border: 1px solid var(--border);
        }
        table {
            width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.9rem;
        }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
        th { color: var(--text-muted); font-weight: 600; }
        .status-drift { color: #f87171; font-weight: 600; }
        .status-stable { color: #34d399; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Student Performance Prediction System</h1>
            <p>End-to-End MLOps Pipeline with MLflow Tracking & Real-Time Data Drift Detection</p>
            <div class="badge-grid">
                <span class="badge">Model: RandomForest</span>
                <span class="badge">MLflow Experiment: Student_Performance_Prediction</span>
                <span class="badge">Monitoring: Population Stability Index (PSI)</span>
                <span class="badge">API Status: Active (Port 5000)</span>
            </div>
        </header>

        <div class="grid">
            <!-- Left Card: Live Inference Form -->
            <div class="card">
                <h2>🔮 Live Student Performance Inference</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="study_time">Weekly Study Time (Hours):</label>
                        <input type="number" id="study_time" name="study_time" step="0.5" min="0" max="30" value="5.0" required>
                    </div>
                    <div class="form-group">
                        <label for="absences">School Absences (Days):</label>
                        <input type="number" id="absences" name="absences" min="0" max="60" value="2" required>
                    </div>
                    <div class="form-group">
                        <label for="previous_score">Previous Exam Score (0 - 100):</label>
                        <input type="number" id="previous_score" name="previous_score" min="0" max="100" value="75.0" required>
                    </div>
                    <div class="form-group">
                        <label for="parental_education">Parental Education:</label>
                        <select id="parental_education" name="parental_education">
                            <option value="Bachelor" selected>Bachelor</option>
                            <option value="High School">High School</option>
                            <option value="Master">Master</option>
                            <option value="PhD">PhD</option>
                            <option value="None">None</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="internet_access">Home Internet Access:</label>
                        <select id="internet_access" name="internet_access">
                            <option value="Yes" selected>Yes</option>
                            <option value="No">No</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">Predict Pass / Fail</button>
                </form>

                <div class="result-box" id="resultContainer" style="display: none;">
                    <span id="predBadge" class="pred-pill">Pass</span>
                    <div class="confidence-meter">Confidence: <span id="confValue" style="color:#fff; font-weight:700;">--</span></div>
                </div>
            </div>

            <!-- Right Card: MLOps Monitoring & Drift Control -->
            <div class="card">
                <h2>🛡️ MLOps Monitoring & Data Drift Control</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 16px;">
                    Simulate production data shifts, calculate Population Stability Index (PSI), and auto-retrain models.
                </p>

                <button id="runDriftBtn" class="btn" style="background: linear-gradient(135deg, #0284c7, #06b6d4);">
                    Execute Data Drift Detection & Retrain
                </button>
                <div id="driftSpinner" style="display:none; margin-top:10px; color:#38bdf8; font-size:0.9rem;">
                    Analyzing PSI distributions and retraining model... please wait...
                </div>

                <div style="margin-top: 20px;">
                    <h3 style="font-size: 1.05rem; margin-bottom: 8px;">Latest Drift Metrics (PSI):</h3>
                    <div id="driftTableContainer">
                        <table>
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>PSI</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="driftTableBody">
                                <tr><td>study_time</td><td>3.05</td><td class="status-drift">Drift Detected</td></tr>
                                <tr><td>absences</td><td>3.68</td><td class="status-drift">Drift Detected</td></tr>
                                <tr><td>previous_score</td><td>1.32</td><td class="status-drift">Drift Detected</td></tr>
                                <tr><td>gender</td><td>0.01</td><td class="status-stable">Stable</td></tr>
                                <tr><td>parental_education</td><td>0.11</td><td class="status-stable">Stable</td></tr>
                                <tr><td>internet_access</td><td>0.12</td><td class="status-stable">Stable</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <div style="margin-top: 24px;">
                    <h3 style="font-size: 1.0rem; margin-bottom: 6px;">REST API Sample (curl):</h3>
                    <div class="api-tester">
curl -X POST http://127.0.0.1:5000/predict \<br>
&nbsp;&nbsp;-H "Content-Type: application/json" \<br>
&nbsp;&nbsp;-d '{"study_time": 5, "absences": 2, "previous_score": 75}'
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const payload = {
                study_time: parseFloat(document.getElementById('study_time').value),
                absences: parseInt(document.getElementById('absences').value),
                previous_score: parseFloat(document.getElementById('previous_score').value),
                parental_education: document.getElementById('parental_education').value,
                internet_access: document.getElementById('internet_access').value
            };

            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            const container = document.getElementById('resultContainer');
            const badge = document.getElementById('predBadge');
            const conf = document.getElementById('confValue');

            container.style.display = 'block';
            badge.innerText = data.prediction;
            badge.className = 'pred-pill pred-' + data.prediction;
            conf.innerText = (data.confidence * 100).toFixed(1) + '%';
        });

        document.getElementById('runDriftBtn').addEventListener('click', async function() {
            const btn = document.getElementById('runDriftBtn');
            const spinner = document.getElementById('driftSpinner');
            btn.disabled = true;
            spinner.style.display = 'block';

            try {
                const res = await fetch('/detect-drift', { method: 'POST' });
                const data = await res.json();
                
                const tbody = document.getElementById('driftTableBody');
                tbody.innerHTML = '';
                
                data.details.forEach(item => {
                    const row = document.createElement('tr');
                    const isDrift = item['Drift Detected'] === 'Yes';
                    row.innerHTML = `
                        <td>${item['Feature Name']}</td>
                        <td>${item['PSI Score'].toFixed(2)}</td>
                        <td class="${isDrift ? 'status-drift' : 'status-stable'}">
                            ${isDrift ? 'Drift Detected' : 'Stable'}
                        </td>
                    `;
                    tbody.appendChild(row);
                });
                alert('Drift detection and automated retraining completed successfully!');
            } catch (err) {
                alert('Drift detection failed: ' + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def dashboard():
    """Interactive visual dashboard for testing and monitoring."""
    return render_template_string(HTML_DASHBOARD_TEMPLATE)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy" if model_exists else "model_missing",
        "service": "student-performance-prediction",
        "model_loaded": model_exists
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Model prediction API.
    Input Format:
    {
        "study_time": 5,
        "absences": 2,
        "previous_score": 75
    }
    
    Output Format:
    {
        "prediction": "Pass",
        "confidence": 0.92
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request payload must be in JSON format."}), 400
        
    data = request.get_json()
    if not data:
        return jsonify({"error": "Empty request body received."}), 400
        
    # Validate required minimal numerical inputs
    required_keys = ["study_time", "absences", "previous_score"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {missing}. Expected at minimum: 'study_time', 'absences', 'previous_score'."
        }), 400
        
    try:
        # Cast numerical features safely
        clean_input = {
            "study_time": float(data["study_time"]),
            "absences": int(data["absences"]),
            "previous_score": float(data["previous_score"])
        }
        
        # Include optional categorical features if present
        if "gender" in data:
            clean_input["gender"] = str(data["gender"])
        if "parental_education" in data:
            clean_input["parental_education"] = str(data["parental_education"])
        if "internet_access" in data:
            clean_input["internet_access"] = str(data["internet_access"])
            
        prediction_result = predict_single(clean_input)
        return jsonify(prediction_result), 200
        
    except Exception as e:
        return jsonify({"error": f"Inference error: {str(e)}"}), 500

@app.route("/detect-drift", methods=["POST"])
def trigger_drift_detection():
    """
    On-demand endpoint to execute data drift detection and automated retraining.
    """
    try:
        drift_res = detect_data_drift()
        if drift_res["overall_drift_detected"]:
            run_drift_pipeline_and_retrain()
            
        return jsonify({
            "status": "success",
            "overall_drift_detected": drift_res["overall_drift_detected"],
            "details": drift_res["report_rows"]
        }), 200
    except Exception as e:
        return jsonify({"error": f"Drift detection pipeline failed: {str(e)}"}), 500

@app.route("/drift-report", methods=["GET"])
def get_drift_report():
    """Returns the latest drift report in JSON format."""
    json_path = os.path.join(REPORTS_DIR, "drift_report.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            report_data = json.load(f)
        return jsonify(report_data), 200
    else:
        return jsonify({"message": "No drift report generated yet."}), 404

if __name__ == "__main__":
    print("\n=======================================================")
    print(" Starting Flask Student Performance API Server")
    print(" Serving on http://127.0.0.1:5000")
    print(" Endpoints:")
    print("   POST /predict       : Make Pass/Fail predictions")
    print("   GET  /              : Interactive Web Dashboard")
    print("   GET  /health        : API Healthcheck")
    print("   POST /detect-drift  : Run drift analysis & retraining")
    print("=======================================================\n")
    # Bind to 127.0.0.1 on port 5000
    app.run(host="127.0.0.1", port=5000, debug=False)
