import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

filepath = "Kraken_OHLCVT/XBTUSD_15with_features.csv"

data = pd.read_csv(filepath)

numerical_features = data.select_dtypes(include=np.number).columns.tolist()
data = data[numerical_features]

data = data.to_numpy()

X = data[:, :-1]
y = data[:, -1]

# Shift labels to make them non-negative (so they fall in range [0, 2])
y = y + 1  # -1 → 0, 0 → 1, 1 → 2
y = np.nan_to_num(y, nan=0).astype(int)

# Reshape X for LSTM (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)


tscv = TimeSeriesSplit(n_splits=5)

window_size = 50

accuracy_scores = []
f1_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Class distribution in training data:")
    unique, counts = np.unique(y_train, return_counts=True)
    print(dict(zip(unique, counts)))

    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    # Convert to dictionary for Keras
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print("Class Weights:", class_weight_dict)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes: Up, Down, No Change
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    accuracy_scores.append(acc)
    f1_scores.append(f1)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

# Print average scores
print(f"\nAverage Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")