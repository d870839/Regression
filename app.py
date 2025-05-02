from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import os
import uuid
from regression_pipeline import run_regression_pipeline  # your core logic

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Store regression results globally for simplicity
regression_results = pd.DataFrame()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['csv']
        if file and file.filename.endswith('.csv'):
            filename = str(uuid.uuid4()) + '.csv'
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            global regression_results
            regression_results = run_regression_pipeline(path)

            return redirect(url_for('dashboard'))

    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if regression_results.empty:
        return redirect(url_for('index'))
    items = sorted(regression_results['price_item'].unique())
    return render_template('dashboard.html', items=items)

@app.route('/predict', methods=['POST'])
def predict():
    item = request.json['item']
    new_price = float(request.json['price'])

    filtered = regression_results[regression_results['price_item'] == item].copy()
    filtered['predicted_change'] = (new_price - 1.0) * 10 * filtered['coefficient_per_dime']
    return jsonify(filtered.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
