import pandas as pd
import numpy as np

# Sample synthetic weather dataset (safe for beginners)
dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")

data = pd.DataFrame({
    "Date": dates,
    "Temperature": np.random.normal(25, 5, 1000)  # fake temperature data
})

data.to_csv("data/weather.csv", index=False)

print("Weather dataset created successfully!")
