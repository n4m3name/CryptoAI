from datetime import datetime
import numpy as np
import pandas as pd
from itertools import tee, product
from collections import deque


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

"""
Keep this filepath for any commits. 
Just label the file correctly and move it into the working directory
"""

# List of coins and times
coins = ['XBTUSD', 'ETHUSD', 'XRPUSD', 'SOLUSD', 'XDGUSD', 'PEPEUSD', 'XCNUSD', 'SUIUSD', 'LTCUSD', 'ADAUSD', 'LINKUSD', 'AAVEUSD', 'ONDOUSD', 'ACHUSD', 'WIFUSD', 'LDOUSD', 'TAOUSD', 'DOTUSD']

times = [1, 5, 15, 30, 60, 240, 720, 1440]

# path generator
paths = (f"Kraken_OHLCVT/{coin}_{time}.csv" for coin in coins for time in times)


# Dataframe generator
framemaker = (pd.read_csv(path) for path in paths)

labels = ["Timestamp", "Open", "High", "Low", "Close", "Value", "Trades"]

# Labels for the columns of the dataframe
column_adder = map(lambda x: x.set_axis(labels, axis=1), framemaker)


#print(df.describe())

# Add the features
feature_adder = map(add_features, column_adder)

# Rename the filepath to the destination filepath
filepaths = (f"coins/{coin}_{time}_with_features.csv" for coin, time in product(coins, times))

# Add the csv with added features to the destination filepath (should be right beside the original file)
deque((df.to_csv(filepath, index=False) for df, filepath in zip(feature_adder, filepaths)), maxlen=0)

# Numpy data object
#data = np.genfromtxt(filepath, delimiter=',', skip_header=0, filling_values=np.nan)