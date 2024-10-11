import pandas as pd
from sqlalchemy import create_engine

# Create a SQLite database connection
engine = create_engine('sqlite:///walmart_sales.db')

# Load the CSV file
df = pd.read_csv('data/Walmart_Sales.csv')

# Group by store and get the last 10 rows for each store
df_last_10 = df.groupby('Store').tail(10).reset_index(drop=True)

# Write the DataFrame to the SQLite database
df_last_10.to_sql('walmart_sales', engine, if_exists='replace', index=False)
print("Data successfully loaded into the SQLite database.")
