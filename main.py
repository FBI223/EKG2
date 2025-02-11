import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import wfdb
from imblearn.over_sampling import SMOTE
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

mitdb_path = "mitdb/"
svdb_path = "svdb/"

# Poprawiona mapa etykiet
label_map = {
    'N': 0,  # Normalny rytm
    'V': 1,  # Pobudzenie komorowe (PVC)
    'A': 2,  # Pobudzenie przedsionkowe (PAC) - MITDB
    'S': 5,  # Pobudzenie nadkomorowe (SVPB) - SVDB
    'L': 3,  # Blok lewej odnogi (LBBB)
    'R': 4   # Blok prawej odnogi (RBBB)
}

num_classes = len(label_map)  # Usunięto dodatkową klasę "Unknown"

def get_class_weights(labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: class_weights[i] for i in range(len(class_weights))}

def resample_ecg_signal(signal, annotation_samples, original_fs, target_fs):
    new_length = int(len(signal) * (target_fs / original_fs))
    resampled_signal = resample(signal, new_length)
    scale_factor = target_fs / original_fs
    resampled_annotations = np.round(np.array(annotation_samples) * scale_factor).astype(int)
    return resampled_signal, resampled_annotations

def get_record_ids(db_path):
    return sorted(list(set(f.split('.')[0] for f in os.listdir(db_path) if f.endswith('.hea'))))

def load_ecg_data(db_path, record_ids, target_fs=360):
    signals, labels, rr_intervals = [], [], []
    for record_id in record_ids:
        record = wfdb.rdrecord(f'{db_path}/{record_id}')
        annotation = wfdb.rdann(f'{db_path}/{record_id}', 'atr')
        signal = record.p_signal[:, 0]
        original_fs = record.fs
        if original_fs != target_fs:
            signal, annotation.sample = resample_ecg_signal(signal, annotation.sample, original_fs, target_fs)
        rr_intervals.extend(np.diff(annotation.sample))
        for i, r in enumerate(annotation.sample[:-1]):
            next_r = annotation.sample[i + 1]
            segment = signal[r:next_r]
            if annotation.symbol[i] in label_map:
                signals.append(segment)
                labels.append(label_map[annotation.symbol[i]])
    return signals, labels, rr_intervals

def determine_optimal_segment_length(rr_intervals_mitdb, rr_intervals_svdb):
    combined_rr_intervals = np.concatenate([rr_intervals_mitdb, rr_intervals_svdb])
    return int(np.median(combined_rr_intervals))

def preprocess_signals(signals, labels, optimal_segment_length):
    filtered_signals, filtered_labels = [], []
    for i in range(len(labels)):
        segment = signals[i]
        if len(segment) >= optimal_segment_length:
            filtered_signals.append(segment[:optimal_segment_length])
        else:
            filtered_signals.append(np.pad(segment, (0, optimal_segment_length - len(segment)), mode='constant'))
        filtered_labels.append(labels[i])
    return np.array(filtered_signals), np.array(filtered_labels)

def build_cnn(input_shape, number_of_classes):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(number_of_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def balance_dataset(X, y):
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled



from imblearn.over_sampling import SMOTE
from collections import Counter

def main():
    # Wczytanie rekordów MITDB
    record_ids = get_record_ids(mitdb_path)
    signals, labels, rr_intervals = load_ecg_data(mitdb_path, record_ids)

    # Wczytanie rekordów SVDB
    record_ids_svdb = get_record_ids(svdb_path)
    signals_svdb, labels_svdb, rr_intervals_svdb = load_ecg_data(svdb_path, record_ids_svdb)

    # Obliczanie optymalnej długości segmentu na podstawie obu baz
    optimal_segment_length = determine_optimal_segment_length(rr_intervals, rr_intervals_svdb)

    # Przetwarzanie sygnałów
    X, y = preprocess_signals(signals + signals_svdb, labels + labels_svdb, optimal_segment_length)

    # Normalizacja danych (globalna)
    X = (X - np.mean(X)) / np.std(X)

    # Podział na zestawy treningowe, walidacyjne i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # Reshape do CNN
    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]

    # Konwersja etykiet do one-hot encoding
    y_train, y_val, y_test = to_categorical(y_train, num_classes=num_classes), to_categorical(y_val, num_classes=num_classes), to_categorical(y_test, num_classes=num_classes)

    # Sprawdzenie liczebności klas przed balansowaniem
    print(f"Before balancing: {Counter(y_train.argmax(axis=1))}")

    # Balansowanie klas za pomocą SMOTE (oversampling)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train.argmax(axis=1))
    y_train = to_categorical(y_train, num_classes=num_classes)
    X_train = X_train.reshape(-1, optimal_segment_length, 1)  # Przywrócenie wymiarów CNN

    # Sprawdzenie liczebności klas po balansowaniu
    print(f"After balancing: {Counter(y_train.argmax(axis=1))}")

    # Obliczenie wag klas (dla modelu)
    class_weights = get_class_weights(y_train.argmax(axis=1))

    # Budowa i trening modelu
    model = build_cnn((optimal_segment_length, 1), num_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=64,
              class_weight=class_weights, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    # Ocena modelu na zbiorze testowym
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
    model.save("ecg_classifier_mitbih_svdb.h5")

if __name__ == "__main__":
    main()
