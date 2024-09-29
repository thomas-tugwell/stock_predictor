import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import json
import time

# Download NLTK data
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Define constants
STOCK_API_KEY = 'your_key'
NEWS_API_KEY = 'your_key' 
STOCK_SYMBOL = 'AAPL'  # Example: Apple Inc.
DATE_FORMAT = '%Y-%m-%d'
query = 'apple'
from_date = '2024-09-20'
to_date = '2024-09-27'

# Get today's date
today = datetime.datetime.now().strftime(DATE_FORMAT)

# Get the date from 2 years ago and 3 months ago
two_years_ago = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime(DATE_FORMAT)
three_months_ago = (datetime.datetime.now() - datetime.timedelta(days=90)).strftime(DATE_FORMAT)

# Function to fetch stock price data
def fetch_stock_prices(symbol, start_date, end_date):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={STOCK_API_KEY}&outputsize=full'
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print("Error fetching stock data")
        return None

    stock_data = data["Time Series (Daily)"]
    prices = []

    for date, values in stock_data.items():
        if start_date <= date <= end_date:
            prices.append({
                "Date": date,
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": int(values["5. volume"])
            })

    return pd.DataFrame(prices)



NEWS_DATA_FILE = 'news_data.json'  # File to store fetched news data

#TODO: update with NewsCatcher API
# Function to fetch and calculate news sentiment with improved storage and caching
def fetch_news_sentiment(query, from_date, to_date):
    # Load existing data from file if available
    if os.path.exists(NEWS_DATA_FILE):
        with open(NEWS_DATA_FILE, 'r') as file:
            news_data = json.load(file)
    else:
        news_data = {}

    # Generate a unique key for the query and date range to check for existing data
    query_key = f"{query}_{from_date}_{to_date}"

    # Check if data is already available for the given date range
    if query_key in news_data:
        print(f"Using cached data for {query} from {from_date} to {to_date}")
        return news_data[query_key]['average_sentiment']

    # If not, fetch new data from NewsAPI
    url = f'https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=popularity&apiKey={NEWS_API_KEY}'
    response = requests.get(url)
    data = response.json()

    if response.status_code != 200 or "articles" not in data:
        print(f"Error fetching news data: {response.status_code}, {data.get('message', 'No message')}")
        return None

    articles = data["articles"]
    sentiment_scores = []

    for article in articles:
        if article.get('description'):
            score = sentiment_analyzer.polarity_scores(article['description'])
            sentiment_scores.append(score['compound'])

    # Calculate the average sentiment
    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    # Store the data in the dictionary for future use
    news_data[query_key] = {
        'query': query,
        'from_date': from_date,
        'to_date': to_date,
        'average_sentiment': avg_sentiment,
        'articles': articles
    }

    # Save the updated news data to file
    with open(NEWS_DATA_FILE, 'w') as file:
        json.dump(news_data, file, indent=4)

    print(f"Fetched and stored new sentiment data for {query} from {from_date} to {to_date}")
    return avg_sentiment


average_sentiment = fetch_news_sentiment(query, from_date, to_date)
print(f"Average Sentiment: {average_sentiment}")

# Fetch stock price data for the past 2 years
stock_df = fetch_stock_prices(STOCK_SYMBOL, two_years_ago, today)
if stock_df is None or stock_df.empty:
    print("Error: No stock data available.")
else:
    print("Stock Prices:")
    print(stock_df.head())
    # Fetch average sentiment for the past 3 months
    #average_sentiment = fetch_news_sentiment(NEWS, three_months_ago, today)
    if average_sentiment is None:
        print("Error: No sentiment data available.")
        average_sentiment = 0  # Use a default sentiment value

    print(f"Average Sentiment for {STOCK_SYMBOL}: {average_sentiment}")

    # Combine stock data with sentiment score
    stock_df['Sentiment'] = average_sentiment

    # Sort the data by date
    stock_df = stock_df.sort_values('Date')

    # Feature Engineering
    stock_df['Price_Change'] = stock_df['Close'].pct_change()
    stock_df.dropna(inplace=True)

    if not stock_df.empty:
        # Normalize the data
        scaler = MinMaxScaler()
        stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']] = scaler.fit_transform(stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']])

        # Create input features for CNN
        def create_features(data, time_steps=10):
            X, y = [], []
            for i in range(time_steps, len(data)):
                X.append(data[i-time_steps:i])
                y.append(data[i, 3])  # Predicting the 'Close' price
            return np.array(X), np.array(y)

        # Prepare features and labels
        features, target = create_features(stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']].values)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Build the CNN model
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))

        # Compile and fit the model
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

        # Predicting and evaluating
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Mean Squared Error: {mse}")

        # Function to predict future stock prices
        def predict_future_prices(days_ahead):
            last_sequence = stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change']].values[-10:]  # 10 days as input
            last_sequence_scaled = last_sequence.reshape(1, last_sequence.shape[0], last_sequence.shape[1])
            future_prices = []
            for _ in range(days_ahead):
                predicted_price = model.predict(last_sequence_scaled)
                future_prices.append(predicted_price[0][0])
                # Update last sequence with the predicted price
                new_entry = np.array([[last_sequence_scaled[0, -1, 0], last_sequence_scaled[0, -1, 1], last_sequence_scaled[0, -1, 2], predicted_price[0][0], last_sequence_scaled[0, -1, 4], last_sequence_scaled[0, -1, 5]]])
                last_sequence_scaled = np.append(last_sequence_scaled[:, 1:, :], new_entry.reshape(1, 1, 6), axis=1)
            return future_prices

        # Predicting next day's, next week's, and next month's stock price
        predicted_1_day = predict_future_prices(1)[-1]
        predicted_1_week = predict_future_prices(7)[-1]
        predicted_1_month = predict_future_prices(30)[-1]
        
        # Output predictions
        print(f"Predicted 1 Day Close Price: {scaler.inverse_transform([[0, 0, 0, predicted_1_day, 0, 0]])[0][3]}")
        print(f"Predicted 1 Week Close Price: {scaler.inverse_transform([[0, 0, 0, predicted_1_week, 0, 0]])[0][3]}")
        print(f"Predicted 1 Month Close Price: {scaler.inverse_transform([[0, 0, 0, predicted_1_month, 0, 0]])[0][3]}")
    else:
        print("Error: Stock DataFrame is empty after processing.")



