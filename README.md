# Sales Forecasting API using Regression
This API provides weekly sales forecasts for a store chain based on historical time series data. It leverages a combination of machine learning and statistical models to make accurate predictions and forecasts. The models are deployed as a Docker container on AWS Elastic Container Service (ECS) and are accessible via a public API endpoint.

## Dataset
- Store - Unique number ID for each store.
- Date - Date of the recorded sales.
- Weekly_Sales - Total sales for the week at the specific store.  
- Holiday_Flag - Indicates whether the week includes a holiday (1 if yes, 0 if no). 
- Temperature - Average temperature for the week (Fahrenheit). 
- Fuel_Price - Fuel price in the store's location (USD per gallon). 
- CPI - Consumer Price Index, reflecting the cost of living for the area. 
- Unemloyment - Unemployment rate in the store's area.

<img width="584" alt="image" src="https://github.com/user-attachments/assets/7475c1cf-be0e-4fda-99b2-2d76f807c91c">
<img width="583" alt="image" src="https://github.com/user-attachments/assets/b9603af1-d2a5-4bae-9db0-5be10bbaadf4">

## Preprocessing 

## Models 
This API combines multiple models to improve accuracy and provide robust predictions:
- **Random Forest** and **XGBoost**: These regression models are both used to predict weekly sales based on features related to each store. The API takes the average of their predictions to provide a final weekly sales estimate.
- **ARIMA**: This statistical model is used for forecasting future sales over a specified period, capturing trends and seasonality in the time series data to generate accurate multi-step forecasts.

## API Endpoints
The API provides the following endpoints:

1. **Predict Sales** (`/predict_sales`)
   - **Method**: `POST`
   - **Description**: Predicts weekly sales for a specified date and store using a combination of Random Forest and XGBoost models.
   - **Input**: JSON object containing store details and input variables like `Date`, `Temperature`, `Fuel_Price`, `CPI`, and `Unemployment`.
   - **Output**: JSON object with predicted sales.

2. **Forecast Sales** (`/forecast_sales`)
   - **Method**: `POST`
   - **Description**: Provides a forecast of weekly sales for a specified number of future weeks using the ARIMA model.
   - **Input**: JSON object containing the `store_id` and the number of `steps` to forecast.
   - **Output**: JSON array with sales forecasts for each future week.

## Example Requests
Use `curl` or HTTP client to make requests to the API.

**Predict Sales:**
```bash
curl -X POST http://<public_ip>/predict_sales \
-H "Content-Type: application/json" \
-d '{
  "Store": 1,
  "Date": "03-11-2012",
  "Holiday_Flag": 0,
  "Temperature": 75.5,
  "Fuel_Price": 3.45,
  "CPI": 238.2,
  "Unemployment": 5.8
}'
```
**Result:**
```json
{
  "prediction": 25000.00
}
```

**Forecast Sales:**
```bash
curl -X POST http://<public_ip>/forecast_sales \
-H "Content-Type: application/json" \
-d '{
  "store_id": 1,
  "steps": 5
}'
```
**Result:**
```json
[
  {"Date": "2023-11-05", "Sales": 25000.00},
  {"Date": "2023-11-12", "Sales": 26000.00},
  {"Date": "2023-11-19", "Sales": 25500.00},
  {"Date": "2023-11-26", "Sales": 26500.00},
  {"Date": "2023-12-03", "Sales": 27000.00}
]
```
