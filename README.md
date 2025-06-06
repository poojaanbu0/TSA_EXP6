### DEVELOPED BY: Pooja A
### REGISTER NO: 212222240072
## DATE:

# Ex.No: 6               HOLT WINTERS METHOD
 

# AIM:
To implement the Holt Winters Method Model using Python.

# ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

# PROGRAM:
```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

# Load the dataset
file_path = 'future_gold_price.csv'  # Update with your actual file path
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Check for non-numeric values in the 'Close' column
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

# Drop any rows with NaN values in the 'Close' column
data = data.dropna(subset=['Close'])

# Resample the data to monthly frequency (beginning of the month)
monthly_data = data['Close'].resample('MS').mean()

# Split the data into train and test sets
train_data = monthly_data[:int(0.9 * len(monthly_data))]  # First 90% for training
test_data = monthly_data[int(0.9 * len(monthly_data)):]   # Last 10% for testing

# Apply the Holt-Winters model without seasonality
fitted_model = ExponentialSmoothing(train_data, trend='mul', seasonal=None).fit()

# Forecast on the test set
test_predictions = fitted_model.forecast(len(test_data))

# Plot the results
plt.figure(figsize=(12, 8))
train_data.plot(legend=True, label='Train')
test_data.plot(legend=True, label='Test')
test_predictions.plot(legend=True, label='Predicted')
plt.title('Train, Test, and Predicted using Holt-Winters')
plt.show()

# Evaluate the model performance
mae = mean_absolute_error(test_data, test_predictions)
mse = mean_squared_error(test_data, test_predictions)
print(f"Mean Absolute Error = {mae}")
print(f"Mean Squared Error = {mse}")

# Fit the model to the entire dataset and forecast the future
final_model = ExponentialSmoothing(monthly_data, trend='mul', seasonal=None).fit()

forecast_predictions = final_model.forecast(steps=12)  # Forecast 12 future periods

# Plot the original and forecasted data
plt.figure(figsize=(12, 8))
monthly_data.plot(legend=True, label='Original Data')
forecast_predictions.plot(legend=True, label='Forecasted Data')
plt.title('Original and Forecasted using Holt-Winters')
plt.show()

```

### OUTPUT:

## TEST_PREDICTION
#![Screenshot 2024-10-07 221222](https://github.com/user-attachments/assets/b190c0bf-c7dc-4c2d-93c3-f0ca77e2ea7f)

### FINAL_PREDICTION
![Screenshot 2024-10-07 221243](https://github.com/user-attachments/assets/3bf4b003-314a-4b2f-ac7d-6d0b7749dec0)


# RESULT:
Thus the program run successfully based on the Holt Winters Method model.
