import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ścieżka do folderu z MIT-BIH
mitdb_path = "mitdb/"

# Pobranie listy wszystkich rekordów
record_ids = [f.split('.')[0] for f in os.listdir(mitdb_path) if f.endswith('.hea')]
record_ids = list(set(record_ids))  # Usunięcie duplikatów
record_ids.sort()

# Wypisanie znalezionych rekordów
print(f"Znalezione rekordy: {record_ids}")

# Pobranie sygnałów EKG
signals = []
labels = []
rr_intervals = []

for record_id in record_ids:
    record = wfdb.rdrecord(f'mitdb/{record_id}')
    annotation = wfdb.rdann(f'mitdb/{record_id}', 'atr')
    signal = record.p_signal[:, 0]
    fs = record.fs

    rr_intervals.extend(np.diff(annotation.sample))  # Obliczanie odstępów RR

    for i, r in enumerate(annotation.sample):
        if i + 1 < len(annotation.sample):
            next_r = annotation.sample[i + 1]
            segment = signal[r:next_r]
            signals.append(segment)
            labels.append(annotation.symbol[i])

# Określenie optymalnej długości segmentu na podstawie mediany odstępów RR
optimal_segment_length = int(np.median(rr_intervals))
print(f"Optymalna długość segmentu: {optimal_segment_length}")

# Przycięcie/pad segmentów do optymalnej długości
filtered_signals = []
filtered_labels = []
label_map = {'N': 0, 'V': 1, 'A': 2, 'L': 3, 'R': 4}

for i in range(len(labels)):
    if labels[i] in label_map:
        segment = signals[i]
        if len(segment) >= optimal_segment_length:
            filtered_signals.append(segment[:optimal_segment_length])
        else:
            pad_width = optimal_segment_length - len(segment)
            filtered_signals.append(np.pad(segment, (0, pad_width), mode='constant'))
        filtered_labels.append(label_map[labels[i]])

X = np.array(filtered_signals)
y = np.array(filtered_labels)

# Normalizacja danych
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Podział na zbiory treningowe, walidacyjne i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

y_train = to_categorical(y_train, num_classes=5)
y_val = to_categorical(y_val, num_classes=5)
y_test = to_categorical(y_test, num_classes=5)

# Definicja modelu CNN 1D
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Trenowanie modelu
model = build_cnn((optimal_segment_length, 1))
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=50, batch_size=32, callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ])

# Ewaluacja modelu
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Macierz pomyłek
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.show()

# Zapis modelu
model.save("ecg_classifier_mitbih.h5")
