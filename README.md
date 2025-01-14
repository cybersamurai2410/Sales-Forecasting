# Sales Forecasting API using Regression with MLOps
This API provides weekly sales forecasts for a store chain based on historical time series data using ETL (Extract, Transform, Load) to leverage machine learning and statistical regression models to deliver accurate predictions. 
The project incorporates MLOps principles to ensure a streamlined and automated workflow for model development including training and hyperparameter tuning, deployment and monitoring. 
Models are trained and logged with MLflow server running on AWS EC2 instance with the artifacts stored in an AWS S3 bucket. 
The API and models are containerized using Docker and deployed by being pushed to Elastic Container Registry (ECR) and orchestarted via AWS Elastic Container Service (ECS); infrastructure provisioning and updates are automated with Python scripts using AWS Boto3 library. 
The CI/CD pipeline is implemented via GitHub Actions, which facilitates continuous automated testing, building and deployment. This architecture demonstrates scalable and automated infrastrutue for machine learning solutions with MLOps. 

**File Structure:**
* `aws_cloud_infra/`
  * `infra.py` - Automates the deployment of AWS ECS infrastructure for running the FastAPI application.
  * `mlflow_infra.py` - Automates the deployment of AWS EC2, RDS, and S3 resources for hosting the MLflow server.
* `workflows/`
  * `main.yml` - Defines the CI/CD pipeline for testing, building, and deploying the application using GitHub Actions.
* `Dockerfile` - Specifies the environment and dependencies for containerizing the FastAPI application.
* `README.md`
* `client.py` - Script for sending API requests to the FastAPI application for predictions.
* `database_loader.py` - Load the sales data from the original CSV dataset into the SQLite dataset.
* `main.py` - FastAPI application file handling prediction endpoints and integrating with MLflow.
* `sales-forecast.ipynb` - Jupyter notebook for the machine learning pipeline including exploratory data analysis, training and evaluation of the sales forecasting models.
* `train.py` - Script to automate training, evaluation, and logging of machine learning models to MLflow
* `unit_tests.py` - Unit tests for the FastAPI application endpoints, including mocking for external dependencies.

**Run Server Commands:**
```bash
fastapi run main.py
```

```bash
uvicorn main:app --reload
```

```bash
mlflow server --host PUBLIC_IP --port 5000
```

## Dataset
- Store - Unique number ID for each store (42 stores total).
- Date - Date of the recorded sales.
- Weekly_Sales - Total sales for the week at the specific store.  
- Holiday_Flag - Indicates whether the week includes a holiday (1 if yes, 0 if no). 
- Temperature - Average temperature for the week (Fahrenheit). 
- Fuel_Price - Fuel price in the store's location (USD per gallon). 
- CPI - Consumer Price Index, reflecting the cost of living for the area. 
- Unemloyment - Unemployment rate in the store's area.

<img width="584" alt="image" src="https://github.com/user-attachments/assets/7475c1cf-be0e-4fda-99b2-2d76f807c91c">
<img width="583" alt="image" src="https://github.com/user-attachments/assets/b9603af1-d2a5-4bae-9db0-5be10bbaadf4">
<img width="478" alt="image" src="https://github.com/user-attachments/assets/d1aff3aa-0641-4fd5-9d39-071fb2fb88c3">

## Preprocessing 
- Convert Date feature to datetime format and indexed for timeseries.
- Feature engineering lagged features of weekly sales capturing temporal dependencies from past two weeks.
- Split Date into day of the week, month and week of the year features for the regression the models to factor temporal dependencies.
- One-hot Encoding applied to the Store feature as binary features for the regression models to consider relationship with stores without assuming ordinality. 

## Models 
**Random Forest** and **XGBoost**:<br>
These tree-based regression models are both trained with hyperparameter tuning to predict weekly sales based on features related to each store. The API takes the average of their predictions to provide a final weekly sales estimate.

<img width="583" alt="image" src="https://github.com/user-attachments/assets/48da8164-f4e6-47b9-9aff-03d9c85189c5">
<img width="565" alt="image" src="https://github.com/user-attachments/assets/2b59dee6-3ab2-4b53-918d-cbe58f6aa09f">
<img width="563" alt="image" src="https://github.com/user-attachments/assets/42921e2e-2fe8-437c-987e-4c1fdd5c134f">

**LSTM (Long Short-Term Memory)**:<br>
This recurrent neural network that captures long-term dependencies when predicting weekly sales for each store. The model was exculed in the API due to overfitting and computational complexity compared to the tree-based models. 

<img width="467" alt="image" src="https://github.com/user-attachments/assets/d854dcbe-053f-4368-8f70-5f181b4a15be">

**ARIMA (Auto-Regressive Integrated Moving Average)**:<br>
This statistical model is used for forecasting future sales over a specified period, capturing trends and seasonality in the time series data to generate multi-step forecasts. The model is univariate focusing on the change in weekly sales instead of the other features. 

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
