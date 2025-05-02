from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import uuid
import pandas as pd
from regression_pipeline import run_regression_pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

regression_results = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    global regression_results

    if request.method == 'POST':
        file = request.files.get('csv_file')
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}.csv")
            file.save(filepath)

            regression_results = run_regression_pipeline(filepath)
            if regression_results.empty:
                return "No significant relationships found. Please upload a different file.", 400

            return redirect(url_for('dashboard'))

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    global regression_results
    if regression_results.empty:
        return redirect(url_for('index'))

    items = sorted(regression_results['price_item'].unique())
    return render_template('dashboard.html', items=items)

@app.route('/predict', methods=['POST'])
def predict():
    global regression_results

    data = request.get_json()
    item = data['item']
    new_price = float(data['price'])

    filtered = regression_results[regression_results['price_item'] == item].copy()
    filtered['predicted_change'] = (new_price - 1.0) * 10 * filtered['coefficient_per_dime']

    filtered = filtered.sort_values(by='predicted_change', ascending=False)
    
    return jsonify(filtered.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
