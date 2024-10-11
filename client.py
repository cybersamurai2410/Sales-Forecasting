import requests
import json

# API endpoint URL
url = "http://localhost:8000/predict"

# Previous date: "26-10-2012"

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
