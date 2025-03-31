import os
import pandas as pd


def test_volatility(filename):
    print("looking at:" + filename)
    try:
        df = pd.read_csv("Kraken_OHLCVT/" + filename)
    except Exception as e:
        return -1
    df.columns = ["Timestamp", "Open", "High", "Low", "Close", "Value", "Trades"]
    df["returns"] = df["Open"].pct_change()
    # Calculate True Range (TR)
    df['TR'] = df['High'] - df['Low']

    atr = df['TR'].rolling(window=14).mean().iloc[-1]  # ATR of the last data point
        
    # Calculate Percent Volatility (ATR / close * 100)
    volatility = (atr / df['Close'].iloc[-1]) * 100

    return volatility

folder_path = "Kraken_OHLCVT"

volatilites = {}

def get_volatility_samples():
    # Populate volatilites dictionary
    for filename in os.listdir(folder_path):
        if "1440" in filename and "USD" in filename and "features" not in filename:
            file_path = os.path.join(folder_path, filename)

            # Check if file contains more than 1000 rows
            if sum(1 for _ in open(file_path)) > 1000:
                volatility = test_volatility(filename)
                if volatility > 0:
                    volatilites[filename] = volatility
    # Sort the dictionary by value (volatility)
    volatilites_sorted = sorted(volatilites.items(), key=lambda item: item[1])
    sample_num = len(volatilites_sorted) // 10
    volatilites_sample = volatilites_sorted[::sample_num]
    # Output top 5 max and min keys with values
    print("10 Samples of Increasing Volatility")
    print(volatilites_sample)
    return volatilites_sample

get_volatility_samples()