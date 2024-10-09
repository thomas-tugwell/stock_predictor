import requests
import pandas as pd
import asyncio
import websockets
import json
from datetime import datetime

# Your Twelve Data API key
API_KEY = '0c00e86997304c678c25271a30899c5b'

# REST API URL for fetching historical data
BASE_URL = 'https://api.twelvedata.com/time_series'

# WebSocket URL for streaming real-time stock data
WS_URL = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={API_KEY}"

# Function to fetch historical data from Twelve Data REST API
def fetch_historical_data(symbol, interval='1min'):
    params = {
        'symbol': symbol,
        'interval': interval,
        'start_date': pd.Timestamp.today().strftime('%Y-%m-%d'),  # Today's date
        'apikey': API_KEY
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if 'values' in data:
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        print(f"Fetched historical data for {symbol}:")
        print(df.head())
        
        # Replace '/' with '_' for filename
        symbol_safe = symbol.replace('/', '_')
        
        # Save the initial historical data to a CSV file
        df.to_csv(f"{symbol_safe}_data.csv", index=False, mode='w')  # Overwrite or create a new file
        return df
    else:
        print("Error fetching historical data.")
        return None

# Function to append real-time data to the historical DataFrame
def append_real_time_data(historical_df, symbol, price, timestamp):
    # Convert the Unix timestamp to a human-readable format and round to the nearest minute
    timestamp = pd.to_datetime(timestamp, unit='s').floor('T')

    # In real-time data, we assume "open", "high", "low", and "close" are all the same as the current price.
    new_data = {
        'datetime': timestamp,
        'open': price,
        'high': price,
        'low': price,
        'close': price
    }

    # Append new data point to the DataFrame
    historical_df = historical_df.append(new_data, ignore_index=True)
    historical_df = historical_df.sort_values('datetime')
    
    # Replace '/' with '_' for filename
    symbol_safe = symbol.replace('/', '_')

    # Append the new data point to the CSV file in the same format as historical data
    pd.DataFrame([new_data]).to_csv(f"{symbol_safe}_data.csv", index=False, header=False, mode='a')
    
    return historical_df

# Function to subscribe to real-time stock data
async def subscribe_to_stocks(websocket, symbols):
    subscription_message = {
        "action": "subscribe",
        "params": {
            "symbols": symbols
        }
    }
    await websocket.send(json.dumps(subscription_message))
    print(f"Subscribed to real-time updates for: {symbols}")

# Function to process the WebSocket message
def process_message(data):
    if "symbol" in data and "price" in data:
        return {
            "symbol": data['symbol'],
            "price": float(data['price']),
            "timestamp": data['timestamp']  # Keep timestamp in seconds
        }
    return None

# Function to stream real-time data and update the historical data
async def stream_real_time_data(symbol, historical_df):
    last_recorded_minute = None  # Track the last minute the data was appended to the CSV

    async with websockets.connect(WS_URL) as websocket:
        # Subscribe to stock symbols
        await subscribe_to_stocks(websocket, symbols=symbol)

        # Continuously receive real-time updates and append to historical data
        while True:
            try:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Received WebSocket message: {data}")

                # Check for valid price data
                processed_data = process_message(data)
                if processed_data:
                    # Extract symbol, price, and timestamp
                    symbol = processed_data['symbol']
                    price = processed_data['price']
                    timestamp = processed_data['timestamp']

                    # Convert the timestamp to datetime for minute tracking
                    current_time = pd.to_datetime(timestamp, unit='s').floor('T')  # Round down to the nearest minute
                    current_minute = current_time.minute

                    # Only append to CSV if the minute has changed
                    if current_minute != last_recorded_minute:
                        # Append new real-time data to the historical DataFrame and CSV file
                        historical_df = append_real_time_data(historical_df, symbol, price, timestamp)
                        print(f"Updated data for {symbol} at {current_time}: Price: {price}")
                        last_recorded_minute = current_minute  # Update the last recorded minute

                else:
                    print("Invalid data format received.")
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed. Attempting to reconnect...")
                break
            except json.JSONDecodeError as e:
                print(f"Error decoding message: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

# Main function to fetch historical data and start real-time data streaming
async def main():
    # Define the cryptocurrency symbol
    symbol = 'ETH/BTC'

    # Step 1: Fetch historical data for the day
    historical_data = fetch_historical_data(symbol, '1min')
    
    if historical_data is not None:
        # Step 2: Stream real-time data and update the historical DataFrame
        await stream_real_time_data(symbol, historical_data)

# Run the main event loop for fetching and streaming data
asyncio.get_event_loop().run_until_complete(main())
