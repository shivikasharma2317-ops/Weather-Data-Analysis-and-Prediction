import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("data/weather.csv")

# Convert date to number
df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = df["Date"].map(pd.Timestamp.toordinal)

# Features and target
X = df[["Date"]]
y = df["Temperature"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy:", round(score, 2))

# Plot
plt.figure(figsize=(10,5))
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.scatter(X_test, predictions, color="red", label="Predicted")
plt.legend()
plt.title("Weather Temperature Prediction")
plt.show()
