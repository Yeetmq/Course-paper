from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model_Occupier = joblib.load('best_model_Occupier.pkl')
model_Investment = joblib.load('best_model_Investment.pkl')

with open('removed_columns.txt', 'r') as f:
    removed_columns = [line.strip() for line in f.readlines()]

def preprocess_data(input_data):

    data = pd.DataFrame([input_data])

    data = data.drop([
        'ID_metro',
        'ID_railroad_station_walk',
        'ID_railroad_station_avto',
        'ID_big_road1',
        'ID_big_road2',
        'ID_railroad_terminal',
        'ID_bus_terminal'
    ], axis=1)

    data = data.drop(columns=removed_columns, errors='ignore', inplace=True)

    data = data.drop('id', axis=1)

    numeric_columns = data.loc[:, data.dtypes!='object'].columns

    for col in numeric_columns:
        data[col] = data[col].fillna(data[col].mean())

    categorical_columns = data.loc[:, data.dtypes == 'object'].columns

    for col in categorical_columns:
        if col != 'timestamp':
            if data[col].nunique() < 5:
                one_hot = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat((data.drop(col, axis=1), one_hot), axis=1)

            else:
                mean_target = data.groupby(col)['full_sq'].mean()
                data[col] = data[col].map(mean_target)

    data['timestamp'] = pd.to_datetime(data['timestamp'])

    data['month'] = data.timestamp.dt.month
    data['year'] = data.timestamp.dt.year

    data = data.sort_values(['timestamp'])

    one_hot = pd.get_dummies(data['year'], prefix='year', drop_first=True)
    data = pd.concat((data.drop('year', axis=1), one_hot), axis=1)
    data = data.replace({True: 1, False: 0})

    one_hot = pd.get_dummies(data['month'], prefix='month', drop_first=True)
    data = pd.concat((data.drop('month', axis=1), one_hot), axis=1)
    data = data.replace({True: 1, False: 0})

    data = data.drop('timestamp', axis=1)

    Owner_Occupier = data[data['product_type_OwnerOccupier'] == 1].copy()
    Investment = data[data['product_type_OwnerOccupier'] == 0].copy()

    return Owner_Occupier, Investment

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    Owner_Occupier, Investment = preprocess_data(input_data)

    prediction_Occupier = model_Occupier.predict(Owner_Occupier)
    prediction_Investment = model_Investment.predict(Investment)

    return jsonify({
        "Owner_Occupier_predictions": prediction_Occupier.tolist(),
        "Investment_predictions": prediction_Investment.tolist()
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)