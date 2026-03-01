from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from preprocess import preprocess_input  # your custom preprocessing function

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/StackModel.joblib')

# Define the original column names (as they appear in the form)
original_columns = [
    'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
    'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

@app.route('/')
def home():
    return render_template('index.html', columns=original_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from form
    input_data = {}
    for col in original_columns:
        value = request.form.get(col)
        # Convert numeric fields appropriately
        if col in ['age', 'duration', 'campaign', 'pdays', 'previous']:
            input_data[col] = int(value)
        elif col in ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']:
            input_data[col] = float(value)
        else:
            input_data[col] = value

    # Convert to DataFrame (single row)
    df_input = pd.DataFrame([input_data])

    # --- Apply preprocessing to get the 32 features ---
    try:
        X_processed = preprocess_input(df_input)  # Must return DataFrame with correct columns
    except Exception as e:
        return render_template('result.html', error=f"Preprocessing error: {str(e)}")

    # Make prediction
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0]  # if classifier

    # Interpret result (adjust based on your target)
    result = "Yes (client subscribed)" if prediction == 1 else "No (client did not subscribe)"
    confidence = max(probability) * 100

    return render_template('result.html', result=result, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)