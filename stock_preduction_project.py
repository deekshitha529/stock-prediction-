import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch historical stock data for Amazon (ticker: AMZN)
amazon_data = yf.download('AMZN', start='2010-01-01', end='2022-01-01')

# Feature engineering: calculating moving averages
amazon_data['MA50'] = amazon_data['Close'].rolling(window=50).mean()
amazon_data['MA200'] = amazon_data['Close'].rolling(window=200).mean()

# Drop rows with missing values
amazon_data.dropna(inplace=True)

# Define features (X) and target variable (y)
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200']
X = amazon_data[features]
y = amazon_data['Close']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Plotting actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test.values, label='Actual Price')
plt.plot(y_test.index, y_pred, label='Predicted Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Prediction example for the next day
last_data_point = X[-1:]
next_day_prediction = model.predict(last_data_point)
print("Predicted Stock Price for the Next Day:", next_day_prediction)
