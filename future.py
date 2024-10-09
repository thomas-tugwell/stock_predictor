import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import time

# Disable GPU (force CPU-only execution)
tf.config.set_visible_devices([], 'GPU')

# Clear any previous TensorFlow session
from tensorflow.keras import backend as K
K.clear_session()

# --- Load the model ---
def load_model(model_name="lstm_model.h5"):
    model = tf.keras.models.load_model(model_name)
    print(f"Model {model_name} loaded successfully.")
    return model

# --- Load new data ---
def load_data_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    print(f"Loaded data: {df.shape[0]} rows")
    return df

# --- Apply feature engineering (moving averages, RSI, etc.) ---
def moving_average(data, window_size):
    return data['close'].rolling(window=window_size).mean()

def rsi(data, window_size=120):  # Adjusted to 2 hours = 120 mins
    delta = data['close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window_size).mean()
    avg_loss = pd.Series(loss).rolling(window=window_size).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    df['close'] = df['close'].astype(float)
    df['ma_5'] = moving_average(df, 5)
    df['ma_10'] = moving_average(df, 10)
    df['rsi_2h'] = rsi(df, window_size=120)  # Use a 2-hour window for RSI
    df = df.dropna(subset=['ma_5', 'ma_10', 'rsi_2h'])
    print(f"Data after feature engineering: {df.shape[0]} rows")
    return df

# --- Prepare LSTM input sequences ---
def prepare_lstm_data(df, sequence_length, scaler):
    features = df[['close', 'ma_5', 'ma_10', 'rsi_2h']].values
    features_scaled = scaler.transform(features)
    
    X = []
    for i in range(sequence_length, len(features_scaled)):
        X.append(features_scaled[i-sequence_length:i])
    
    return np.array(X)

# --- Make predictions ---
def make_predictions(model, X, scaler):
    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform([[p, 0, 0, 0] for p in predictions[:, 0]])[:, 0]
    return predicted_prices

# --- Live plotting with both past and future predictions ---

def live_plot_real_time_predictions(csv_file, model, sequence_length=60, steps_ahead=120, interval=10):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(14, 8))  # Create a figure and axis

    # Initialize lists to store the running predictions and corresponding timestamps
    past_predictions = []
    past_prediction_times = []

    future_predictions = []
    future_prediction_times = []

    while True:
        # Fetch the latest data
        df = load_data_from_csv(csv_file)
        
        if len(df) > sequence_length:
            # Apply feature engineering
            df = add_features(df)
            
            # Prepare the data for prediction
            scaler = MinMaxScaler()
            scaler.fit(df[['close', 'ma_5', 'ma_10', 'rsi_2h']])
            X_real_time = prepare_lstm_data(df, sequence_length, scaler)
            
            # Predict the next steps (e.g., next 10 minutes)
            current_prediction = make_predictions(model, X_real_time[-steps_ahead:], scaler)
            
            # Append only the most recent minute's prediction to the past predictions
            most_recent_time = df['datetime'].iloc[-1] + pd.Timedelta(minutes=1)
            past_predictions.append(current_prediction[0])  # Add the predicted price
            past_prediction_times.append(most_recent_time)  # Add the timestamp

            # Append the future predictions
            future_times = pd.date_range(most_recent_time, periods=len(current_prediction)+1, freq="T")[1:]
            future_predictions = current_prediction.tolist()
            future_prediction_times = future_times.tolist()

            # Filter to show only the last 30 minutes of actual prices
            last_30_minutes = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(minutes=30)]
            last_120_minutes = df[df['datetime'] >= df['datetime'].max() - pd.Timedelta(minutes=480)]
            
            # Clear previous plots to refresh the figure
            ax.clear()
            
            # Plot actual prices for the last 240 minutes
            ax.plot(last_120_minutes['datetime'], last_120_minutes['close'], label="Actual Price", color="blue")
            
            # Plot past predictions (for each minute predicted)
            ax.plot(past_prediction_times, past_predictions, label="Past Predictions", color="green", linestyle="--")
            
            # Plot future predictions (extend beyond actual data)
            ax.plot(future_prediction_times, future_predictions, label="Future Predictions", color="red", linestyle="--")
            
            # Set the x-axis format to show hours and minutes
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            
            # Ensure ticks are only placed every 5 minutes
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=60))  # Tick marks every 5 minutes
            ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))  # Minor tick marks every 1 minute (optional)

            # Adjust the grid lines and tick settings for clarity
            ax.grid(True, which='major', axis='both', linestyle='--', alpha=0.5)  # Grid for major ticks
            
            # Ensure ticks don't overlap
            plt.xticks(rotation=45)

            # Labels and title
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.set_title(f"Real-Time Stock Price Prediction (Last 30 minutes)")
            ax.legend()
            
            # Draw the updated plot
            plt.draw()
            
            # Pause for a short time before refreshing (adjust interval as needed)
            plt.pause(interval)
        else:
            print("Not enough data for prediction, waiting for more data...")
            time.sleep(interval)  # Wait before checking the file again

# --- Main Function ---
if __name__ == "__main__":
    # Load the saved model
    model = load_model("lstm_model.h5")

    # CSV file that is continuously updated
    csv_file = 'ETH_BTC_data.csv'
    
    # Live plot real-time predictions
    live_plot_real_time_predictions(csv_file, model, sequence_length=60, steps_ahead=120, interval=30)
