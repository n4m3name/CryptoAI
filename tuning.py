import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import os
import preprocessor
import volatility_tester

volatility_samples = volatility_tester.get_volatility_samples()

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(units=256, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=128, return_sequences=False),
        Dropout(0.2),
        Dense(256, activation='tanh'),
        Dropout(0.2),
        Dense(128, activation='tanh'),
        Dense(1)
    ])
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model

# Function to create sequences for LSTM
def create_sequences(X, y, seq_length):
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i : i + seq_length])
        labels.append(y[i + seq_length])
    return np.array(sequences), np.array(labels)

volatilites = []
maes = []

print(volatility_samples)

for f, v in volatility_samples:
    K.clear_session()  # Clear previous session
    print(f"Processing file: {f}")
    
    data = preprocessor.process_file("Kraken_OHLCVT/" + f)
    print(f"Initial data shape for {f}: {data.shape}")
    
    # Remove non-numeric columns
    data.drop(columns=["Timestamp"], errors="ignore", inplace=True)
    print(f"Data shape after dropping non-numeric columns: {data.shape}")
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"NaN count before dropping: {data.isna().sum().sum()}")
    
    data.dropna(inplace=True)
    print(f"Data shape after dropping NaNs: {data.shape}")
    
    if data.empty:
        print(f"Data after preprocessing is empty for {f}. Skipping...")
        continue
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Ensure correct feature count
    if data_scaled.shape[1] < 4:
        print(f"Not enough features in {f}. Skipping...")
        continue
    
    X_raw = data_scaled[:, :-1]  # All columns except target
    y_raw = data_scaled[:, 3]  # Assuming column 3 is the closing price
    
    sequence_length = 50
    num_features = X_raw.shape[1]
    print(f"Number of features for {f}: {num_features}")
    
    # Create sequences
    X, y = create_sequences(X_raw, y_raw, sequence_length)
    
    if X.shape[0] == 0:
        print(f"Insufficient data after sequence processing for {f}. Skipping...")
        continue
    
    # Train-validation-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    val_split_idx = int(len(X_train) * 0.8)
    X_train_final, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
    y_train_final, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
    
    print(f"Training samples: {len(X_train_final)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Build a fresh model
    model = build_model((sequence_length, num_features))
    
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train_final, y_train_final,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test MAE: {test_mae:.4f}")
    maes.append(test_mae)
    volatilites.append(v)

    


# Plot volatility vs mae
plt.figure(figsize=(8, 5))
plt.plot(volatilites, maes, label="Training Loss", color='blue')
plt.xlabel("Volatility")
plt.ylabel("Test MAE")
plt.title("Volatility vs Test MAE")
plt.legend()
plt.grid()
plt.show()
