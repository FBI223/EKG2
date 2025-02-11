import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import wfdb
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight


mitdb_path = "mitdb/"
svdb_path = "svdb/"
#label_map = {'N': 0, 'V': 1, 'A': 2, 'L': 3, 'R': 4}  # Dodajemy "Unknown" jako len(label_map)
label_map = {
    'N': 0,  # Normalny rytm
    'V': 1,  # Pobudzenie komorowe (PVC)
    'A': 2,  # Pobudzenie przedsionkowe (PAC) - MITDB
    'S': 5,  # Pobudzenie nadkomorowe (SVPB) - SVDB
    'L': 3,  # Blok lewej odnogi (LBBB)
    'R': 4   # Blok prawej odnogi (RBBB)
}

num_classes = len(label_map) + 1  # Nowa klasa "Unknown"



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








def load_svdb_data(svdb_path, target_fs=360):
    """
    Wczytuje dane z MIT-BIH Supraventricular Arrhythmia Database (SVDB) i dostosowuje częstotliwość do 360 Hz.
    :param svdb_path: Ścieżka do folderu SVDB
    :param target_fs: Docelowa częstotliwość próbkowania (domyślnie 360 Hz)
    :return: Przetworzone sygnały i adnotacje
    """
    record_ids = get_record_ids(svdb_path)
    signals, labels, rr_intervals, unknown_signals = load_ecg_data(svdb_path, record_ids, target_fs)
    return signals, labels, rr_intervals, unknown_signals






def load_ecg_data(db_path, record_ids, target_fs=360):
    signals, labels, rr_intervals = [], [], []
    unknown_signals = []

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
                labels.append(annotation.symbol[i])
            else:
                unknown_signals.append(segment)

    return signals, labels, rr_intervals, unknown_signals

def determine_optimal_segment_length(rr_intervals):
    return int(np.median(rr_intervals))

def preprocess_signals(signals, labels, optimal_segment_length, unknown_signals):
    filtered_signals, filtered_labels = [], []
    for i in range(len(labels)):
        segment = signals[i]
        if len(segment) >= optimal_segment_length:
            filtered_signals.append(segment[:optimal_segment_length])
        else:
            filtered_signals.append(np.pad(segment, (0, optimal_segment_length - len(segment)), mode='constant'))
        filtered_labels.append(label_map[labels[i]])

    for segment in unknown_signals:
        if len(segment) >= optimal_segment_length:
            filtered_signals.append(segment[:optimal_segment_length])
        else:
            filtered_signals.append(np.pad(segment, (0, optimal_segment_length - len(segment)), mode='constant'))
        filtered_labels.append(len(label_map))  # Klasa "Unknown"

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


def main():


    record_ids = get_record_ids(mitdb_path)
    signals, labels, rr_intervals, unknown_signals = load_ecg_data(mitdb_path, record_ids)
    optimal_segment_length = determine_optimal_segment_length(rr_intervals)

    svdb_signals, svdb_labels, svdb_rr_intervals, svdb_unknown_signals = load_svdb_data(svdb_path)

    X, y = preprocess_signals(signals + svdb_signals, labels + svdb_labels, optimal_segment_length, unknown_signals + svdb_unknown_signals)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]
    y_train, y_val, y_test = to_categorical(y_train, num_classes=num_classes), to_categorical(y_val, num_classes=num_classes), to_categorical(y_test, num_classes=num_classes)

    model = build_cnn((optimal_segment_length, 1), num_classes)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    y_pred_classes = np.where(np.max(y_pred, axis=1) < 0.4, len(label_map), y_pred_classes)  # Próg 0.5 na "Unknown"

    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa klasa")
    plt.show()

    model.save("ecg_classifier_mitbih_svdb_unknown.h5")

if __name__ == "__main__":
    main()


