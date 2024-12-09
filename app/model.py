from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder



class ModelHandler:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
    
    def load_model(self, path: str):
        import joblib
        return joblib.load(path)

    def predict_Occupier(self, data: pd.DataFrame) -> np.ndarray:
        preprocessed_data = self.preprocess_data_Occupier(data)
        return self.model.predict(preprocessed_data)
    
    def predict_Investment(self, data: pd.DataFrame) -> np.ndarray:
        print(data.columns)
        print(data.head())
        return self.model.predict(data)

    def preprocess_data_Occupier(self, input_data: pd.DataFrame) -> pd.DataFrame:
        
        with open('removed_columns.txt', 'r') as f:
            removed_columns = [line.strip() for line in f.readlines()]

        data = pd.DataFrame(input_data)

        data = data.drop([
            'ID_metro',
            'ID_railroad_station_walk',
            'ID_railroad_station_avto',
            'ID_big_road1',
            'ID_big_road2',
            'ID_railroad_terminal',
            'ID_bus_terminal'
        ], axis=1)

        data = data.drop(columns=removed_columns, errors='ignore')

        data = data.drop('id', axis=1)

        numeric_columns = data.loc[:, data.dtypes!='object'].columns

        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].mean())

        categorical_columns = data.loc[:, data.dtypes == 'object'].columns

        label_encoder = LabelEncoder()

        # Применяем Label Encoding для категориальных столбцов
        for col in categorical_columns:
            if col != 'timestamp':
                # Применяем LabelEncoder
                data[col] = label_encoder.fit_transform(data[col])
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        data['month'] = data.timestamp.dt.month
        data['year'] = data.timestamp.dt.year

        data = data.sort_values(['timestamp'])
        
        data['year'] = label_encoder.fit_transform(data['year'])

        data['year'] = label_encoder.fit_transform(data['month'])

        data = data.drop('timestamp', axis=1)

        Owner_Occupier = data[data['product_type'] == 1].copy()

        return Owner_Occupier
    
    def preprocess_data_Investment(self, input_data: pd.DataFrame) -> pd.DataFrame:
        
        with open('removed_columns.txt', 'r') as f:
            removed_columns = [line.strip() for line in f.readlines()]

        data = pd.DataFrame(input_data)

        data = data.drop([
            'ID_metro',
            'ID_railroad_station_walk',
            'ID_railroad_station_avto',
            'ID_big_road1',
            'ID_big_road2',
            'ID_railroad_terminal',
            'ID_bus_terminal'
        ], axis=1)

        data = data.drop(columns=removed_columns, errors='ignore')

        data = data.drop('id', axis=1)

        numeric_columns = data.loc[:, data.dtypes!='object'].columns

        for col in numeric_columns:
            data[col] = data[col].fillna(data[col].mean())

        categorical_columns = data.loc[:, data.dtypes == 'object'].columns

        label_encoder = LabelEncoder()

        # Применяем Label Encoding для категориальных столбцов
        for col in categorical_columns:
            if col != 'timestamp':
                # Применяем LabelEncoder
                data[col] = label_encoder.fit_transform(data[col])
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        data['month'] = data.timestamp.dt.month
        data['year'] = data.timestamp.dt.year

        data = data.sort_values(['timestamp'])
        
        data['year'] = label_encoder.fit_transform(data['year'])

        data['year'] = label_encoder.fit_transform(data['month'])

        data = data.drop('timestamp', axis=1)

        print(data.head())

        Investment = data[data['product_type'] == 0].copy()


        return Investment
    