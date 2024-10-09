import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# API key for TwelveData
API_KEY = '0c00e86997304c678c25271a30899c5b'

# Function to fetch data from TwelveData API
def fetch_historical_data(symbol, start_date, end_date, interval='1min'):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&start_date={start_date}&end_date={end_date}&apikey={API_KEY}&outputsize=5000"
    response = requests.get(url)
    data = response.json()
    if 'values' in data:
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        return df
    else:
        print(f"Error fetching data: {data}")
        return None

# Function to fetch data in chunks and store in CSV
def fetch_data_in_chunks(symbol, start_date, end_date, chunk_size_days=5, output_csv='historical_data.csv'):
    current_start = start_date

    # Open CSV file and write headers
    with open(output_csv, 'w') as f:
        f.write('datetime,open,high,low,close,volume\n')

    call_count = 0

    # Continue fetching data in chunks until we reach the end date
    while current_start < end_date:
        current_end = current_start + timedelta(days=chunk_size_days)

        # Ensure we don't go beyond the end date
        if current_end > end_date:
            current_end = end_date

        print(f"Fetching data from {current_start.date()} to {current_end.date()}")
        df_chunk = fetch_historical_data(symbol, current_start.date(), current_end.date(), interval='1min')

        if df_chunk is not None:
            df_chunk.to_csv(output_csv, mode='a', header=False, index=False)

        # Move the start date forward by the chunk size
        current_start = current_end + timedelta(days=1)

        # Track the number of API calls
        call_count += 1

        # Throttle if we exceed the rate limit (8 calls per minute)
        if call_count % 8 == 0:
            print("Rate limit reached. Sleeping for 60 seconds...")
            time.sleep(61)  # Sleep for 60 seconds to avoid exceeding the API rate limit

# Main function to fetch data
if __name__ == "__main__":
    symbol = 'ETH/USD'
    # Going back one year from today
    end_date = datetime.today()
    start_date = end_date - timedelta(days=1095)
    
    # Fetch data and store in CSV
    fetch_data_in_chunks(symbol, start_date, end_date, chunk_size_days=5, output_csv='historical_data.csv')
