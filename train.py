import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
from mlflow.models import infer_signature
from statsmodels.tsa.arima.model import ARIMA

# --- DATA LOADING ---
data = pd.read_csv("data/Walmart_Sales.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Feature engineering
data['Lag_1_Week_Sales'] = data.groupby('Store')['Weekly_Sales'].shift(1)
data['Lag_2_Week_Sales'] = data.groupby('Store')['Weekly_Sales'].shift(2)
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['WeekOfYear'] = data['Date'].dt.isocalendar().week
data['Year'] = data['Date'].dt.year

# Fill missing lagged sales values
data.bfill(inplace=True)

# One-hot encode Store column
data = pd.get_dummies(data, columns=['Store'], drop_first=True)

# Drop non-numeric and unused columns
data = data.drop(columns=['Date'])

# --- TRAIN-TEST SPLIT ---
features = [col for col in data.columns if col != 'Weekly_Sales']
X = data[features]
y = data['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- FEATURE SCALING ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for inference
joblib.dump(scaler, "scaler.joblib") # /content/scaler.joblib

# --- MODEL TRAINING AND EVALUATION ---
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# Set up MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("Sales Forecasting Experiment")

# Start an MLflow run
with mlflow.start_run(run_name="Training Run"):
    # Log hyperparameters for XGBoost
    xgb_params = {'n_estimators': 100, 'random_state': 42}
    mlflow.log_params(xgb_params)

    # Train XGBoost model
    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train)

    # Log hyperparameters for Random Forest
    rf_params = {'n_estimators': 100, 'random_state': 42}
    mlflow.log_params(rf_params)

    # Train Random Forest model
    model_rf = RandomForestRegressor(**rf_params)
    model_rf.fit(X_train, y_train)

    # Evaluate both models
    xgboost_scores = evaluate_model(y_test, model_xgb.predict(X_test))
    rf_scores = evaluate_model(y_test, model_rf.predict(X_test))

    # Log evaluation metrics
    for metric, value in xgboost_scores.items():
        mlflow.log_metric(f"xgboost_{metric}", value)
    for metric, value in rf_scores.items():
        mlflow.log_metric(f"rf_{metric}", value)

    # Log averaged final prediction metrics
    final_predictions = (model_xgb.predict(X_test) + model_rf.predict(X_test)) / 2
    final_metrics = evaluate_model(y_test, final_predictions)
    for metric, value in final_metrics.items():
        mlflow.log_metric(f"final_{metric}", value)

    # Log the trained models as MLflow artifacts
    xgb_signature = infer_signature(X_train, model_xgb.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model_xgb,
        artifact_path="xgboost_model",
        registered_model_name="XGB-Sales-Forecasting",
        signature=xgb_signature,
        input_example=X_train,
    )

    rf_signature = infer_signature(X_train, model_rf.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model_rf,
        artifact_path="randomforest_model",
        registered_model_name="RF-Sales-Forecasting",
        signature=rf_signature,
        input_example=X_train,
    )

    # Log feature scaling or transformation
    mlflow.log_artifact("models/scaler.joblib")

    # Save raw predictions to a DataFrame
    predictions_df = pd.DataFrame({
        "True Values": y_test.tolist(),
        "XGBoost Predictions": model_xgb.predict(X_test).tolist(),
        "Random Forest Predictions": model_rf.predict(X_test).tolist(),
        "Final Predictions (Averaged)": final_predictions.tolist()
    })

    # Save the DataFrame to a CSV file
    predictions_path = "data/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    # Log predictions as an artifact
    mlflow.log_artifact(predictions_path, artifact_path="predictions")

# Set up MLflow experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow server
mlflow.set_experiment("Sales Forecasting Experiment")

# Load and preprocess data
data = pd.read_csv("/Walmart_Sales.csv")
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data = data.set_index(['Store', 'Date'])
data.index = data.index.set_levels(data.index.levels[1].to_period('W-FRI'), level='Date')

# Function to train ARIMA model and log with MLflow
def train_arima_with_mlflow(store_data, store_id, order=(5, 1, 0)):
    store_sales = store_data['Weekly_Sales']

    with mlflow.start_run(run_name=f"ARIMA_Store_{store_id}"):
        # Train ARIMA model
        print(f"Training ARIMA for Store {store_id} with order {order}...")
        model = ARIMA(store_sales, order=order)
        arima_model = model.fit()

        # Log parameters
        mlflow.log_param("store_id", store_id)
        mlflow.log_param("order", order)

        # Log metrics
        mlflow.log_metric("aic", arima_model.aic) # AIC (Akaike Information Criterion)
        mlflow.log_metric("bic", arima_model.bic) # BIC (Bayesian Information Criterion)

        # Save and log the model as an artifact
        model_path = f"arima_model_store_{store_id}.joblib"
        joblib.dump(arima_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models/arima")

        print(f"ARIMA training completed for Store {store_id}. AIC: {arima_model.aic}, BIC: {arima_model.bic}")

# Train ARIMA model for each store
stores = data.index.get_level_values('Store').unique()

for store in stores:
    # Filter data for the current store
    store_data = data.xs(store, level='Store')

    # Train and log the ARIMA model
    train_arima_with_mlflow(store_data, store_id=store)

print("ARIMA training completed for all stores.")
