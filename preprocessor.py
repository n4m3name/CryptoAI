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

def label_data(array):
    np.append(array, np.zeros((len(array), 1)), axis=1)
    for i in range(len(array)-2):
        change = (array[i+1][4] - array[i][4]) / array[i][4]
        if change > 0.004:
            array[i][-1] = 2
        elif change > -0.004:
            array[i][-1] = 1
        else:
            array[i][-1] = 0
    
  

lines = get_last_lines("../Kraken_OHLCVT/XBTUSD_15.csv")
lines = convert_date(lines)
print(lines[:2])
label_data(lines)
print(lines[:2])