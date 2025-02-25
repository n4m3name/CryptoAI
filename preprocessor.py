
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
    date

def convert_date(array): #john
    times = array[:, -1]
    new_arr = np.array

get_last_lines("Kraken_OHLCVT/XBTUSD_15.csv")