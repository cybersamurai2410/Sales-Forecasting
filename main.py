from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from sqlalchemy import create_engine
from mangum import Mangum

# Load the trained model
model = joblib.load('models/xgb_model-tuned.joblib')
model_rf = joblib.load('models/rf_model-tuned.joblib')

# Create a SQLite database connection
engine = create_engine('sqlite:///walmart_sales.db')

# Define the input model using Pydantic
class SalesInput(BaseModel):
    Store: int
    Date: str
    Holiday_Flag: int
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float

app = FastAPI()
handler = Mangum(app)

def get_previous_sales(store):
    # Query the database for the last two weeks of sales for the given store
    query = f"""
    SELECT Date, Weekly_Sales
    FROM walmart_sales
    WHERE Store = {store}
    LIMIT -2
    """
    store_sales = pd.read_sql(query, engine)

    # Handle cases where there may not be enough data
    previous_week_sales = store_sales.iloc[0]['Weekly_Sales'] if len(store_sales) > 0 else 0
    two_weeks_ago_sales = store_sales.iloc[1]['Weekly_Sales'] if len(store_sales) > 1 else 0

    return previous_week_sales, two_weeks_ago_sales

# Function to apply feature engineering
def apply_feature_engineering(input_data):
    # Convert input data to a DataFrame
    df = pd.DataFrame([input_data])

    # Convert Date from string to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

    # Retrieve lagged sales data
    df['Lag_1_Week_Sales'], df['Lag_2_Week_Sales'] = get_previous_sales(df['Store'][0])

    # Extract date features (day of the week, month, week of the year)
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['Year'] = df['Date'].dt.year

    for i in range(1, 46):
        df[f'Store_{i}'] = 0
    df[f'Store_{df["Store"].iloc[0]}'] = 1

    df = df.drop(columns=['Date', 'Store'])

    return df

@app.post("/predict_sales")
async def predict_sales(input_data: SalesInput):
    # Convert the Pydantic input data to dictionary 
    input_dict = input_data.model_dump()

    # Apply feature engineering to get lagged features, date features, and one-hot encoded stores
    input_features = apply_feature_engineering(input_dict)

    # Output sales predictions
    prediction_xb = model.predict(input_features)
    prediction_rf = model_rf.predict(input_features)

    # Average between model predictions 
    prediction = (prediction_xb[0] + prediction_rf[0]) / 2

    # Insert row to SQL database with input_data and prediction
    input_dict['Weekly_Sales'] = prediction
    features = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    df_input = pd.DataFrame([input_dict])[features]
    df_input.to_sql('walmart_sales', engine, if_exists='append', index=False)
    
    return {"prediction": round(prediction, 2)}

@app.post("/forecast_sales")
async def predict_sales_arima(store_id: int, steps: int = 3):

    # Load the saved ARIMA model for the store
    model = joblib.load(f'models/forecast_models/arima_model_store_{store_id}.joblib')

    # Forecast the next 'steps' weeks
    predictions = model.forecast(steps=steps)  

    # Convert the forecast to a DataFrame for easier formatting
    predicted_sales_df = predictions.to_frame(name='Sales').reset_index()
    predicted_sales_df.columns = ['Date', 'Sales']

    # Format the date and sales values
    predicted_sales_df['Date'] = predicted_sales_df['Date'].dt.strftime('%d-%m-%Y')
    predicted_sales_df['Sales'] = predicted_sales_df['Sales'].apply(lambda x: f"{x:.2f}")

    # Convert the DataFrame to a list of dictionaries for JSON response
    forecast = predicted_sales_df.to_dict(orient='records')

    return {"predictions": forecast}

# fastapi run main.py
# uvicorn main:app --reload
