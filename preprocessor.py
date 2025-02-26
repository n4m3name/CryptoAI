from datetime import datetime
import numpy as np
import pandas as pd

"""
Keep this filepath for any commits. 
Just label the file correctly and move it into the working directory
"""
filepath = "Kraken_OHLCVT/XBTUSD_15.csv"

# This is the threshold for determining if a coin went up. I forget what number we wanted to use.
threshold = 0.004

# Dataframe
df = pd.read_csv(filepath)

# Labels for the columns of the dataframe
df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Value", "Trades"]

print(df.describe())

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
    df["SMA_10"] = df["Close"].rolling(window=10).mean()  # 10-period moving average
    df["SMA_50"] = df["Close"].rolling(window=50).mean()  # 50-period moving average
    df["RSI_14"] = compute_rsi(df["Close"]) # 14-period relative strength index
    df["Return"] = df["Close"].pct_change() # Percent change from last close

    # Bollinger Bands
    df["Middle_Band"] = df["Close"].rolling(window=20).mean()
    df["Upper_Band"] = df["Middle_Band"] + (df["Close"].rolling(window=20).std() * 2)
    df["Lower_Band"] = df["Middle_Band"] - (df["Close"].rolling(window=20).std() * 2)

    df["Return_Signal"] = df["Return"].apply(
        lambda x: 1 if x > threshold else (0 if 0 <= x <= threshold else -1)
    )

    add_datetime_features(df)

    return df

# Add the features
df = add_features(df)

# Rename the filepath to the destination filepath
filepath = filepath.removesuffix(".csv") + "with_features.csv"

# Add the csv with added features to the destination filepath (should be right beside the original file)
df.to_csv(filepath, index=False)

# Numpy data object
data = np.genfromtxt(filepath, delimiter=',', skip_header=0, filling_values=np.nan)