import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import wfdb
from scipy.signal import resample
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

ICENTIA11K_PATH = "D:/EKG/BAZA/icentia11k-single-lead-continuous-raw-electrocardiogram-dataset-1.0"
MITDB_PATH = "mitdb/"
SVDB_PATH = "svdb/"

# Optimal segment length: 270

LABEL_MAP = {
    'N': 0,  # Normalny rytm
    'V': 1,  # Pobudzenie komorowe (PVC)
    'A': 2,  # Pobudzenie przedsionkowe (PAC) - MITDB
    'S': 3,  # Pobudzenie nadkomorowe (SVPB) - SVDB
    'L': 4,  # Blok lewej odnogi (LBBB)
    'R': 5   # Blok prawej odnogi (RBBB)
}

NUM_CLASSES = len(LABEL_MAP)



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
            if annotation.symbol[i] in LABEL_MAP:
                signals.append(segment)
                labels.append(LABEL_MAP[annotation.symbol[i]])
    return signals, labels, rr_intervals

def determine_optimal_segment_length(all_intervals):
    combined_rr_intervals = np.concatenate(all_intervals)
    return int(np.median(combined_rr_intervals))

def preprocess_signals(signals, labels, optimal_segment_length):
    filtered_signals, filtered_labels = [], []

    for i in range(len(labels)):
        segment = signals[i]

        # 🔹 Jeśli klasa to 'S', używamy większego okna czasowego
        if labels[i] == 3:  # Klasa S (SVPB)
            target_length = optimal_segment_length * 2  # Dłuższe okno dla S
        else:
            target_length = optimal_segment_length

        # Przycinanie lub padding do odpowiedniego rozmiaru
        if len(segment) >= target_length:
            segment = segment[:target_length]
        else:
            segment = np.pad(segment, (0, target_length - len(segment)), mode='constant')

        # Resampling do jednolitego rozmiaru
        segment = resample(segment, optimal_segment_length)

        filtered_signals.append(segment)
        filtered_labels.append(labels[i])

    return np.array(filtered_signals), np.array(filtered_labels)

def build_cnn(input_shape, number_of_classes):
    model = models.Sequential([
        layers.Conv1D(64, kernel_size=11, strides=1, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=7, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=5, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.1),

        layers.LSTM(64, return_sequences=False),  # Dodanie warstwy LSTM
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(number_of_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def count_annotations(db_path):
    """Zlicza liczbę wystąpień każdej etykiety w zbiorze danych"""
    label_counts = Counter()

    record_ids = get_record_ids(db_path)

    for record_id in record_ids:
        annotation_file = os.path.join(db_path, record_id)
        try:
            annotation = wfdb.rdann(annotation_file, 'atr')  # Wczytanie adnotacji
            label_counts.update(annotation.symbol)  # Zliczanie etykiet
        except Exception as e:
            print(f"❌ Błąd podczas przetwarzania {record_id}: {e}")

    return label_counts


def main():
    # 🔹 Ustawienia TensorFlow dla optymalnej pracy na GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    '''
    label_counts = count_annotations(ICENTIA11K_PATH)
    print("📊 Liczność każdej etykiety w zbiorze Icentia11k:")
    for label, count in label_counts.most_common():
        print(f"{label}: {count}")
    '''

    # Wczytanie rekordów MITDB
    record_ids = get_record_ids(MITDB_PATH)
    signals, labels, rr_intervals = load_ecg_data(MITDB_PATH, record_ids)

    # Wczytanie rekordów SVDB
    record_ids_svdb = get_record_ids(SVDB_PATH)
    signals_svdb, labels_svdb, rr_intervals_svdb = load_ecg_data(SVDB_PATH, record_ids_svdb)

    # Obliczanie optymalnej długości segmentu na podstawie obu baz
    optimal_segment_length = determine_optimal_segment_length([rr_intervals, rr_intervals_svdb])
    print('Optimal segment length:', optimal_segment_length)

    # Przetwarzanie sygnałów
    X, y = preprocess_signals(signals + signals_svdb, labels + labels_svdb, optimal_segment_length)

    # Normalizacja danych (globalna)
    X = (X - np.mean(X)) / np.std(X)

    # Podział na zestawy treningowe, walidacyjne i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

    # Reshape do CNN
    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]

    # Konwersja etykiet do one-hot encoding z dodatkową klasą "Unknown"
    y_train, y_val, y_test = to_categorical(y_train, num_classes=NUM_CLASSES), to_categorical(y_val, num_classes=NUM_CLASSES), to_categorical(y_test, num_classes=NUM_CLASSES)

    # Definicja callbacków (Early Stopping + Reduce LR)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True, monitor='val_loss', min_delta=0.001
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=2, verbose=1, min_lr=1e-5
    )

    # Budowa i trening modelu
    model = build_cnn((optimal_segment_length, 1), NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Ocena modelu na zbiorze testowym
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    # Zapis modelu
    model.save("ecg_classifier_mitbih_svdb.h5")

    # 🔹 Predykcja modelu na zbiorze testowym
    y_pred = model.predict(X_test)
    y_pred_max_confidence = np.max(y_pred, axis=1)  # Maksymalne prawdopodobieństwo
    y_pred_classes = np.argmax(y_pred, axis=1)  # Klasa o najwyższym prawdopodobieństwie

    # Prawdziwe klasy
    y_true = np.argmax(y_test, axis=1)

    # Aktualizacja macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(NUM_CLASSES))

    # 🔹 Dodaj nazwę dla nowej klasy "Unknown"
    class_labels = list(LABEL_MAP.keys())

    # 🔹 Zaktualizuj macierz pomyłek w Pandas
    df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    df_cm.to_csv("confusion_matrix.csv", index=True)
    print("✅ Macierz pomyłek zapisana do 'confusion_matrix.csv'")

    # 🔹 Wizualizacja i zapis jako obraz
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa klasa")
    plt.title("Macierz Pomyłek")

    # Zapis jako plik PNG
    plt.savefig("confusion_matrix.png")
    print("✅ Macierz pomyłek zapisana do 'confusion_matrix.png'")

    # Pokazanie wykresu w notebooku (opcjonalne)
    plt.show()

    # 🔹 Dodatkowa analiza – histogram pewności predykcji
    plt.figure(figsize=(8, 4))
    plt.hist(y_pred_max_confidence, bins=20, edgecolor='black')
    plt.xlabel("Maksymalne prawdopodobieństwo predykcji")
    plt.ylabel("Liczba próbek")
    plt.title("Histogram pewności predykcji")
    plt.savefig("confidence_histogram.png")
    print("✅ Histogram pewności zapisany do 'confidence_histogram.png'")
    plt.show()


if __name__ == "__main__":
    main()
