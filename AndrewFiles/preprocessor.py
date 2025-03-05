from datetime import datetime
import numpy as np
import pandas as pd

# This is the threshold for determining if a coin went up. I forget what number we wanted to use.
threshold = 0.004

def add_datetime_features(df):
    """
    Input: Original dataframe
    Output: Dataframe appended with the features: [Weekday, Day, Month, Year, Time of Day (as TOD)]
    """
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s")  # Convert to datetime
    
    df["Weekday"] = df["Timestamp"].dt.weekday  # 0 = Monday, 6 = Sunday
    df["Day"] = df["Timestamp"].dt.day
    df["Month"] = df["Timestamp"].dt.month
    df["Year"] = df["Timestamp"].dt.year
    df["TOD"] = df["Timestamp"].dt.hour + df["Timestamp"].dt.minute / 60  # Time of day in hours

    return df

def compute_rsi(series, period=14):
    """
    Input: Series of closing prices
    Output: The relative strength index over 14 periods
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_features(df):
    """
    Input: Dataframe without features
    Output: Dataframe with features
    """
    # Calculate the percentage change between the current row and the next row
    df["Return"] = df["Close"].pct_change(periods=-1)  # Use periods=-1 for next row

    # Shift the Return column back by one row to align with the current row
    df["Return"] = df["Return"].shift(1)

    # Calculate the Return_Signal based on the shifted Return column
    df["Return_Signal"] = df["Return"].apply(
        lambda x: 1 if x > threshold else (0 if 0 <= x <= threshold else -1)
    )

    # Add other features (SMA, RSI, Bollinger Bands, etc.)
    df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-period moving average
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-period moving average
    df["RSI_14"] = compute_rsi(df["Close"])  # 14-period relative strength index

    # Bollinger Bands
    df["Middle_Band"] = df["Close"].rolling(window=20).mean()
    df["Upper_Band"] = df["Middle_Band"] + (df["Close"].rolling(window=20).std() * 2)
    df["Lower_Band"] = df["Middle_Band"] - (df["Close"].rolling(window=20).std() * 2)

    # Add datetime features
    add_datetime_features(df)

    return df

def process_file(input_filepath, output_filepath):
    """
    Process a CSV file and save the output with added features.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_filepath)

    # Labels for the columns of the dataframe
    df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Value", "Trades"]

    # Add the features
    df = add_features(df)

    # Save the processed DataFrame to the output filepath
    df.to_csv(output_filepath, index=False)

    print(f"Processed file saved to: {output_filepath}")