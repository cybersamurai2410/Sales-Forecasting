# Sales Forecasting API using Regression with MLOps
This API provides weekly sales forecasts for a store chain based on historical time series data leveraging machine learning and statistical regression models to deliver accurate predictions. The project incorporates MLOps principles to ensure a streamlined and automated end-to-end workflow for data ingestion using ETL and model development including training and hyperparameter tuning, deployment and monitoring. Models are trained and logged with MLflow server running on AWS EC2 instance with the model versions, experiment runs and artifacts stored in AWS S3 bucket and tracked in MySQL database hosted in AWS RDS for monitoring data drift and performance metrics. The API and models are containerized using Docker and deployed by being pushed to Elastic Container Registry (ECR) and orchestrated via AWS Elastic Container Service (ECS); infrastructure provisioning and updates are automated with Python scripts using AWS Boto3 library. Monitoring and alerting for application health, performance and resource utilization are enabled via AWS CloudWatch to allow real-time observability. The CI/CD pipeline is implemented via GitHub Actions, which facilitates continuous automated testing, building and deployment. This architecture demonstrates scalable and automated infrastructure for machine learning solutions with MLOps. 

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
mlflow server --host 127.0.0.1 --port 5000
```

```bash
mlflow server --backend-store-uri mysql+pymysql://<username>:<password>@<rds-endpoint>/<db-name> --default-artifact-root s3://<bucket-name>/ --host 0.0.0.0 --port 5000
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

## Continuous Integration and Continuous Deployment (CI/CD) GitHub Actions Workflow
* **Automated Testing** - Ensures the code is functional and meets quality standards by running unit tests before deployment. This is handled in the `test` job, where dependencies are installed and tests are executed using `pytest`. If the tests fail, the workflow halts, preventing defective code from being deployed.
* **Steps** - Organizes the workflow into sequential jobs to handle testing, infrastructure provisioning and deployment in an orderly manner. The workflow is divided into three jobs; `test`, `deploy-infra`, and `build-and-deploy`. Each job depends on the success of the previous one to ensure proper execution.
* **AWS Infrastructure** - Automates the creation and updating of AWS resources for deploying the application and supporting MLOps workflows. The `deploy-infra` job runs the `infra.py` and `mlflow_infra.py` scripts to provision or update AWS resources like ECS, ECR, S3, and EC2 using the Boto3 library.
* **Docker Image** - Packages the application and its dependencies into a container for consistent deployment across environments. The `build-and-deploy` job builds the Docker image, tags it with the current commit SHA (secure hashing algorithm), pushes it to AWS ECR and updates the AWS ECS with the new image.
* **Secrets Management** - Protects sensitive data like AWS credentials from being exposed in the repository. AWS credentials are stored in GitHub Secrets (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) and accessed securely in the `deploy-infra` and `build-and-deploy` jobs.
* **Monitoring** - Ensures system reliability and performance by tracking application metrics and logs using AWS CloudWatch handled via AWS console.

## Further Work


## MLflow UI
<img width="1115" alt="Screenshot 2025-01-09 143431" src="https://github.com/user-attachments/assets/5a6dd112-1480-44c7-95aa-376b566810a3" />
<img width="1354" alt="Screenshot 2025-01-09 143355" src="https://github.com/user-attachments/assets/fce1eddf-6b45-40c7-9fc4-ca248a2a74a6" />
<img width="1321" alt="Screenshot 2025-01-09 143459" src="https://github.com/user-attachments/assets/bfa796be-ac6c-4677-a46e-65e6af930fa0" />
<img width="1344" alt="Screenshot 2025-01-09 143533" src="https://github.com/user-attachments/assets/092c12c7-64cd-481e-9b77-ab891f036bbc" />
<img width="1340" alt="Screenshot 2025-01-09 143606" src="https://github.com/user-attachments/assets/7da5c6d0-ce59-48e9-bc6d-423d2151125f" />
