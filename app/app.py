from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
from model import ModelHandler

app = Flask(__name__)

model_Occupier = ModelHandler(model_path='label_model_Occupier.pkl')
model_Investment = ModelHandler(model_path='label_model_Investment.pkl')

with open('removed_columns.txt', 'r') as f:
    removed_columns = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # If POST from form
    if request.form:
        input_data = request.form.get('data')
    else:  # If POST from API
        input_data = request.get_json()
    
    try:
        input_data = pd.DataFrame([eval(input_data)])
    except Exception as e:
        return jsonify({"error": "Invalid input data format", "details": str(e)})

    input_data.replace('NA', pd.NA, inplace=True)

    if input_data['product_type'][0] == 'Investment':
        Investment = model_Investment.preprocess_data_Investment(input_data)
        prediction_Investment = model_Investment.predict_Investment(Investment)

        return jsonify({
        "Investment_predictions": np.exp(prediction_Investment).tolist()
    })
    else:
        Owner_Occupier = model_Occupier.preprocess_data_Occupier(input_data)
        prediction_Occupier = model_Occupier.predict_Occupier(Owner_Occupier)

        return jsonify({
            "Owner_Occupier_predictions": np.exp(prediction_Occupier).tolist()
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
