import pytest
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock

# Create a test client for FastAPI
client = TestClient(app)

@patch("main.mlflow")  # Mock the `mlflow` object in `main.py`
def test_predict_sales(mock_mlflow):
    """
    Test the /predict_sales endpoint with valid input.
    Mock MLflow interactions to isolate the test from external dependencies.
    """
    # Mock MLflow behaviors
    mock_mlflow.start_run.return_value = MagicMock()
    mock_mlflow.log_metrics.return_value = None
    mock_mlflow.log_params.return_value = None

    # Valid input data for the endpoint
    input_data = {
        "Store": 1,
        "Date": "01-01-2022",
        "Holiday_Flag": 0,
        "Temperature": 20.0,
        "Fuel_Price": 2.0,
        "CPI": 100.0,
        "Unemployment": 5.0
    }

    # Call the endpoint
    response = client.post("/predict_sales", json=input_data)

    # Assertions to verify the response and behavior
    assert response.status_code == 200  # Ensure the endpoint returns success
    assert "prediction" in response.json()  # Ensure the response contains a "prediction"

    # Verify that mocked MLflow methods were called
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_metrics.assert_called()
    mock_mlflow.log_params.assert_called()

@patch("main.mlflow")  # Mock the `mlflow` object in `main.py`
def test_predict_sales_arima(mock_mlflow):
    """
    Test the /forecast_sales endpoint with valid input.
    Mock MLflow interactions to isolate the test.
    """
    # Mock MLflow behaviors
    mock_mlflow.start_run.return_value = MagicMock()
    mock_mlflow.log_params.return_value = None
    mock_mlflow.log_metrics.return_value = None
    mock_mlflow.log_artifact.return_value = None

    # Valid parameters for the ARIMA forecast
    store_id = 1
    steps = 3

    # Call the endpoint
    response = client.post(f"/forecast_sales?store_id={store_id}&steps={steps}")

    # Assertions to verify the response and behavior
    assert response.status_code == 200  # Ensure the endpoint returns success
    assert "predictions" in response.json()  # Ensure the response contains "predictions"

    # Verify that mocked MLflow methods were called
    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_params.assert_called()
    mock_mlflow.log_metrics.assert_called()

def test_predict_sales_invalid_input():
    """
    Test the /predict_sales endpoint with invalid input.
    """
    # Invalid input data (Store should be an integer)
    input_data = {
        "Store": "abc",  # Invalid data type
        "Date": "01-01-2022",
        "Holiday_Flag": 0,
        "Temperature": 20.0,
        "Fuel_Price": 2.0,
        "CPI": 100.0,
        "Unemployment": 5.0
    }

    # Call the endpoint
    response = client.post("/predict_sales", json=input_data)

    # Assertions to verify the response
    assert response.status_code == 422  # Unprocessable Entity due to validation error

def test_predict_sales_arima_invalid_store_id():
    """
    Test the /forecast_sales endpoint with an invalid store_id.
    """
    # Invalid store_id parameter
    store_id = "abc"  # Invalid data type
    steps = 3

    # Call the endpoint
    response = client.post(f"/forecast_sales?store_id={store_id}&steps={steps}")

    # Assertions to verify the response
    assert response.status_code == 422  # Unprocessable Entity due to validation error

def test_predict_sales_missing_fields():
    """
    Test the /predict_sales endpoint with missing required fields.
    """
    # Missing required fields
    input_data = {
        "Store": 1,  # Only Store and Date are provided
        "Date": "01-01-2022"
    }

    # Call the endpoint
    response = client.post("/predict_sales", json=input_data)

    # Assertions to verify the response
    assert response.status_code == 422  # Unprocessable Entity due to validation error

def test_predict_sales_arima_invalid_steps():
    """
    Test the /forecast_sales endpoint with invalid steps parameter.
    """
    # Invalid steps parameter (negative value)
    store_id = 1
    steps = -1  # Invalid value

    # Call the endpoint
    response = client.post(f"/forecast_sales?store_id={store_id}&steps={steps}")

    # Assertions to verify the response
    assert response.status_code == 422  # Unprocessable Entity due to validation error

# Command to run tests: `pytest tests/unit_tests.py`
