import csv
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd


def get_last_lines(name, n=5000): #john
    with open(name) as f:
        lines = deque(csv.reader(f), maxlen=n)
    return np.array(lines, dtype=float)

def time_to_row(time): #john
    dt = datetime.utcfromtimestamp(time)
    return [dt.weekday(), dt.day, dt.month, dt.year, dt.hour, dt.minute]

def convert_date(array): #john
    times = array[:, 0]
    new_arr = np.array([time_to_row(i) for i in times])
    return np.concatenate((array, new_arr), axis=1)

lines = get_last_lines("Kraken_OHLCVT/XBTUSD_15.csv")
lines = convert_date(lines)
print(lines[:2])