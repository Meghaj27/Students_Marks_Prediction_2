from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained Linear Regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'linear_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Performance Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background: radial-gradient(circle at top left, #1e1b4b, #0f172a, #020617);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px 16px;
        }
        .container {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 560px;
            padding: 36px 32px;
        }
        .header {
            text-align: center;
            margin-bottom: 28px;
        }
        .header h1 {
            font-size: 1.85rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }
        .header p {
            color: #94a3b8;
            font-size: 0.95rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 18px;
        }
        @media (min-width: 480px) {
            .form-grid {
                grid-template-columns: 1fr 1fr;
            }
            .full-width {
                grid-column: span 2;
            }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            background-color: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        .btn-submit {
            grid-column: span 1;
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            margin-top: 10px;
            transition: transform 0.15s ease, opacity 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        }
        @media (min-width: 480px) {
            .btn-submit {
                grid-column: span 2;
            }
        }
        .btn-submit:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55);
        }
        .btn-submit:active {
            transform: scale(0.98);
        }
        .result-box {
            margin-top: 28px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid #6366f1;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #a5b4fc;
            margin-bottom: 4px;
        }
        .result-score {
            font-size: 2rem;
            font-weight: 800;
            color: #38bdf8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Student Score Predictor</h1>
            <p>Enter academic parameters to estimate performance score</p>
        </div>
        
        <form action="/predict" method="POST" class="form-grid">
            <div class="form-group">
                <label for="hours_studied">Hours Studied</label>
                <input type="number" id="hours_studied" name="hours_studied" step="0.1" min="0" max="24" placeholder="e.g. 7.0" required>
            </div>
            
            <div class="form-group">
                <label for="previous_scores">Previous Scores</label>
                <input type="number" id="previous_scores" name="previous_scores" step="0.1" min="0" max="100" placeholder="e.g. 85.0" required>
            </div>
            
            <div class="form-group">
                <label for="extracurricular">Extracurriculars</label>
                <select id="extracurricular" name="extracurricular" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="sleep_hours">Sleep Hours</label>
                <input type="number" id="sleep_hours" name="sleep_hours" step="0.1" min="0" max="24" placeholder="e.g. 8.0" required>
            </div>
            
            <div class="form-group full-width">
                <label for="sample_papers">Sample Papers Practiced</label>
                <input type="number" id="sample_papers" name="sample_papers" step="1" min="0" max="50" placeholder="e.g. 5" required>
            </div>
            
            <button type="submit" class="btn-submit">Predict Estimated Score</button>
        </form>

        {% if prediction_text %}
            <div class="result-box">
                <div class="result-title">Predicted Score</div>
                <div class="result-score">{{ prediction_text }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features matching exact feature names required by the model:
        # ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']
        input_df = pd.DataFrame([{
            'Hours Studied': float(request.form['hours_studied']),
            'Previous Scores': float(request.form['previous_scores']),
            'Extracurricular Activities': int(request.form['extracurricular']),
            'Sleep Hours': float(request.form['sleep_hours']),
            'Sample Question Papers Practiced': int(request.form['sample_papers'])
        }])
        
        prediction = model.predict(input_df)[0]
        formatted_score = f"{prediction:.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_score)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
