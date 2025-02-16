import os
import wfdb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt, resample
from biosppy.signals import ecg
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.signal import butter, filtfilt
from scipy.signal import iirnotch
import pywt



# 📂 Foldery z danymi MITDB i SVDB
MITDB_PATH = "mitdb/"
SVDB_PATH = "svdb/"

# 🔹 Docelowa częstotliwość próbkowania
TARGET_FS = 360
SEGMENT_LENGTH = 200  # Długość segmentu w próbkach (QRS w środku)

# 🔹 Mapowanie etykiet
LABEL_MAP_ORIGINAL = {'N': 0, 'V': 1, 'A': 2, 'S': 3, 'L': 4, 'R': 5}
LABEL_MAP = {'N': 0, 'V': 1, 'S': 2}
LABEL_MAP_MITDB = {'N': 0, 'V': 1, 'A': 2, 'L': 3, 'R': 4}
LABEL_NAMES = list(LABEL_MAP.keys())  # Kolejność klas
NUM_CLASSES = len(LABEL_MAP)




import numpy as np
import pywt
from scipy.signal import butter, filtfilt, sosfilt, iirnotch

def bandpass_filter(signal, fs, lowcut=0.5, highcut=50, order=4):
    """📌 Filtr pasmowo-przepustowy (0.5–50 Hz) do usunięcia zakłóceń mięśniowych i drgań."""
    nyq = 0.5 * fs
    if lowcut >= highcut or highcut >= nyq:
        raise ValueError("Niepoprawne wartości filtracji pasmowo-przepustowej: lowcut < highcut < Nyquist")

    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfilt(sos, signal)

def notch_filter(signal, fs, freq=50, quality_factor=30):
    """📌 Filtr Notch do usunięcia zakłóceń sieciowych (np. 50 Hz lub 60 Hz)."""
    nyq = 0.5 * fs
    if freq >= nyq:
        raise ValueError("Częstotliwość Notch musi być mniejsza niż Nyquist")

    w0 = freq / nyq
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)

def highpass_filter(signal, fs, lowcut=0.5, order=4):
    """📌 Filtr górnoprzepustowy (usuwa drift bazowy poniżej 0.5 Hz)."""
    nyq = 0.5 * fs
    if lowcut >= nyq:
        raise ValueError("Częstotliwość odcięcia highpass musi być mniejsza niż Nyquist")

    low = lowcut / nyq
    sos = butter(order, low, btype='highpass', output='sos')
    return sosfilt(sos, signal)

def wavelet_denoising(signal, wavelet='db6', level=5):
    """📌 Usuwa szum mięśniowy za pomocą DWT (Dekompozycja falkowa)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs_thresh, wavelet)

def filter_ecg_2(signal, fs):
    """📌 Kompleksowa filtracja sygnału EKG:
        - Pasmo 0.5–50 Hz
        - Usunięcie 50 Hz (lub 60 Hz)
        - Eliminacja driftu bazowego
        - Usunięcie szumu mięśniowego falkami
    """
    signal = bandpass_filter(signal, fs)
    signal = notch_filter(signal, fs)
    signal = highpass_filter(signal, fs)
    signal = wavelet_denoising(signal)
    return signal






### 🔥 **1. Filtracja sygnału (redukcja szumów)**
def filter_ecg(signal, fs=TARGET_FS):
    """📌 Filtracja pasmowo-przepustowa 0.5–50 Hz, usunięcie zakłóceń mięśniowych"""
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 50 / nyq
    b, a = butter(4, [low, high], btype='bandpass')
    return filtfilt(b, a, signal)




def visualize_qrs_peak(signal):
    """
    Rysuje i zapisuje interpolowany segment EKG z naniesionymi adnotacjami.

    :param signal: Interpolowany sygnał EKG (1D numpy array).
    :param annotations: Lista indeksów adnotacji po interpolacji.
    :param segment_id: Numer segmentu, do nazwy pliku.
    :param patient_name: Nazwa pacjenta do personalizacji plików.
    """

    plt.figure(figsize=(10, 4))
    plt.plot(signal, color="b", linewidth=1, label="fragment sygnału")

    plt.xlabel("Próbki")
    plt.ylabel("Znormalizowana wartość")
    plt.title(f"Interpolowany segment EKG")
    plt.legend()

    # ✅ Wyświetlenie wykresu na ekranie
    plt.show()





### 🔥 **2. Resampling sygnału**
def resample_ecg_signal(signal, annotation_samples, original_fs, target_fs=TARGET_FS):
    """🔄 Resampling sygnału do docelowej częstotliwości"""
    new_length = int(len(signal) * (target_fs / original_fs))
    resampled_signal = resample(signal, new_length)
    scale_factor = target_fs / original_fs
    resampled_annotations = np.round(np.array(annotation_samples) * scale_factor).astype(int)
    return resampled_signal, resampled_annotations


### 🔥 **3. Wczytywanie i przetwarzanie danych**
def load_ecg_data(db_path, record_ids):
    signals, labels = [], []

    for record_id in record_ids:
        record = wfdb.rdrecord(f'{db_path}/{record_id}')
        annotation = wfdb.rdann(f'{db_path}/{record_id}', 'atr')
        signal = record.p_signal[:, 0]  # Pobranie 1. odprowadzenia
        original_fs = record.fs  # Oryginalna częstotliwość próbkowania

        # 🔹 Resampling do TARGET_FS
        if original_fs != TARGET_FS:
            signal, annotation.sample = resample_ecg_signal(signal, annotation.sample, original_fs, TARGET_FS)

        # 🔹 Filtracja sygnału
        signal = filter_ecg(signal, TARGET_FS)

        # 🔹 Segmentacja QRS w środku
        for i, r in enumerate(annotation.sample):
            if annotation.symbol[i] in LABEL_MAP:
                start = max(0, r - SEGMENT_LENGTH // 2)
                end = min(len(signal), r + SEGMENT_LENGTH // 2)

                segment = signal[start:end]

                if  len(segment) != SEGMENT_LENGTH and len(segment) / SEGMENT_LENGTH > 0.75:
                    segment = np.pad(segment, (0, SEGMENT_LENGTH - len(segment)), mode='edge')
                    visualize_qrs_peak(segment)
                elif len(segment) < SEGMENT_LENGTH:
                    continue


                signals.append(segment)
                labels.append(LABEL_MAP[annotation.symbol[i]])

    return np.array(signals), np.array(labels)


### 🔥 **4. Tworzenie modelu CNN+LSTM**
def build_cnn_lstm(input_shape, num_classes):
    model = models.Sequential([

        layers.Masking(mask_value=0, input_shape=(SEGMENT_LENGTH, 1)),

        layers.Conv1D(64, kernel_size=11, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(128, kernel_size=7, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(256, kernel_size=5, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        layers.LSTM(64, return_sequences=False),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model


### 🔥 **5. Trening modelu i generowanie statystyk**
def train_model():
    # Wczytanie rekordów MITDB i SVDB
    #record_ids = sorted([f.split('.')[0] for f in os.listdir(MITDB_PATH) if f.endswith('.hea')])
    #signals_mitdb, labels_mitdb = load_ecg_data(MITDB_PATH, record_ids)

    record_ids_svdb = sorted([f.split('.')[0] for f in os.listdir(SVDB_PATH) if f.endswith('.hea')])
    signals_svdb, labels_svdb = load_ecg_data(SVDB_PATH, record_ids_svdb)

    # Połączenie zbiorów
    #X, y =  signals_mitdb, labels_mitdb


    # Połączenie zbiorów
    X, y =  signals_svdb, labels_svdb


    # Połączenie zbiorów
    #X, y = np.concatenate((signals_mitdb, signals_svdb)), np.concatenate((labels_mitdb, labels_svdb))

    # Normalizacja
    X = (X - np.mean(X)) / np.std(X)

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train, random_state=42)

    # Reshape dla CNN
    X_train, X_val, X_test = X_train[..., np.newaxis], X_val[..., np.newaxis], X_test[..., np.newaxis]
    y_train, y_val, y_test = to_categorical(y_train, NUM_CLASSES), to_categorical(y_val, NUM_CLASSES), to_categorical(y_test, NUM_CLASSES)

    # 📌 CALLBACKS
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-5)

    # Budowa i trening modelu
    model = build_cnn_lstm((SEGMENT_LENGTH, 1), NUM_CLASSES)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=32, callbacks=[early_stopping, reduce_lr])

    # Ewaluacja modelu
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 🔹 Statystyki
    report = classification_report(y_true, y_pred_classes, target_names=LABEL_NAMES)
    print("\n📊 Statystyki modelu:\n", report)

    # 🔹 Macierz pomyłek
    cm = confusion_matrix(y_true, y_pred_classes)
    print("🔹 Specyficzność (TNR):", cm.diagonal() / cm.sum(axis=1))

    # Zapis modelu
    model.save("ecg_classifier.h5")


if __name__ == "__main__":
    train_model()
