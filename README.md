# Sales Forecasting using Regression
Timeseries sales forecasting of store chain predicting weekly sales for each store using regression models. 

## Example Requests
Use `curl` or HTTP client to make requests to the API.

<br><br>
***/predict_sales***
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
<br><br>
***/forecast_sales***
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
