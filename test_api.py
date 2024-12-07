import pandas as pd
import requests
import simplejson as json

# Загрузим данные с NaN
test_data = pd.read_csv('data/test.csv')

# Преобразуем данные в формат JSON, используя simplejson (оно поддерживает NaN)
test_json = test_data.to_dict(orient='records')

# Сериализуем данные в JSON с использованием simplejson
json_data = json.dumps(test_json, ignore_nan=True)

# Адрес сервиса
url = 'http://127.0.0.1:5000/predict'

# Отправляем запрос
response = requests.post(url, json=json_data)

# Проверяем статус ответа
if response.status_code == 200:
    predictions = response.json()
    print("Owner Occupier Predictions:", predictions['Owner_Occupier_predictions'])
    print("Investment Predictions:", predictions['Investment_predictions'])
else:
    print("Ошибка при отправке запроса:", response.status_code)

