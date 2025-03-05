import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
import csv
import numpy as np
from collections import deque
#import train test split
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from tensorflow.keras.layers import Input



# The size of the dataset to work with
size = 10000


# The number of time steps in each example
timesteps = 60

# The size of each batch
batch_size = 32

# Number of Epochs
epochs = 100


def tuple_generator(x_data, y_data, timesteps):
    ''' Input: A numpy array of Time-series Data
        Yields: A batch of Training and testing data '''

    i = 0
    y_data = y_data[timesteps-1:]
    while True:
        x_batch = np.stack([x_data[j:j+timesteps] for j in range(i, min(batch_size + i, len(x_data) - i))])
        y_batch = y_data[i:i+batch_size]
        i += batch_size
        if i >= len(x_data):
            i = 0
        yield x_batch, y_batch



def example_gen(data, timesteps):
    """ Inputs: A numpy array and a given timesteps
        Yields: an example for the neural network
        Depreciated"""
    i = 0
    while True:
        batch = np.array([data[j:j+timesteps] for j in range(i, min(batch_size + i, len(data) - i))])
        yield batch
        i = (i + min(batch_size, len(data) - i)) % len(data)





def target_gen(data, timesteps):
    """ Inputs: A numpy array and a given timesteps
        Yields: a target for the neural network
        Depreciated"""
    data = data[timesteps-1:]
    i = 0
    while True:
        yield data[i:i+timesteps]
        i = (i + min(batch_size, len(data) - i)) % len(data)


def get_column_names(name):
    """ Input: The name of a csv file
        Returns: The column names of the csv file"""
    with open(name) as f:
        reader = csv.reader(f)
        return next(reader)

def get_last_lines(name, n=5000):
    """ Input: The name of a csv file and the number of lines to read
        Returns: The last n lines of the csv file"""
    with open(name) as f:
        lines = deque(csv.reader(f), maxlen=n+1)
    return np.array(lines)[:-1]

def build_generators(name, timesteps):
    """ Input: The name of a csv file and the number of timesteps
        Returns: Training and Testing generators, the shape of the training and testing data"""
    columns = get_column_names(name)
    data = get_last_lines(name, size)
    frame = pd.DataFrame(data, columns=columns)
    frame.pop('Timestamp')
    frame = frame.astype(float)
    target = frame.pop('Return_Signal')
    print(frame.dtypes)
    xtr, xte, ytr, yte = train_test_split(frame, target, test_size=0.2, shuffle=False) 
    train_shape = xtr.shape
    train_gen = tuple_generator(xtr.to_numpy(), ytr.to_numpy(), timesteps)
    test_gen = tuple_generator(xte.to_numpy(), yte.to_numpy(), timesteps)
    test_shape = xte.shape
    return  train_gen, test_gen, train_shape, test_shape


train_gen, test_gen, train_shape, test_shape = build_generators("drive/MyDrive/Kraken Coins/XBTUSD_15_with_features.csv", timesteps)