import requests
import json

ip = "3.10.52.202" # AWS ECS public URL 

# API endpoint URL 8000 (fastapi) | 80 (docker)
# url = "http://localhost:80/predict_sales"
url = f"http://{ip}/predict_sales"

# Sample input data
input_data = {
    "Store": 1,
    "Date": "03-11-2012", 
    "Holiday_Flag": 0,
    "Temperature": 75.5,
    "Fuel_Price": 3.45,
    "CPI": 238.2,
    "Unemployment": 5.8
}

# Send POST request to the API
response = requests.post(url, json=input_data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    prediction = result["prediction"]
    print(f"Predicted sales: ${prediction:.2f}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

# API endpoint URL for forecast_sales
# url_forecast = "http://localhost:80/forecast_sales"
url_forecast = f"http://{ip}/forecast_sales"

params_forecast = {
    "store_id": 1,
    "steps": 5
}

response_forecast = requests.post(url_forecast, params=params_forecast)

if response_forecast.status_code == 200:
    result_forecast = response_forecast.json()
    predictions = result_forecast.get("predictions", [])

    if predictions:
        print(f"Sales forecast for the next {params_forecast['steps']} weeks:")
        for prediction in predictions:
            date = prediction["Date"]
            sales = prediction["Sales"]
            print(f"Date: {date}, Predicted Sales: ${sales}")
    else:
        print("No predictions found.")
else:
    print(f"Error: {response_forecast.status_code}")
    print(response_forecast.text)
