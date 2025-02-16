import os
import wfdb
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from scipy.signal import resample
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, sosfilt, iirnotch
import seaborn as sns


# 📂 Foldery z danymi MITDB i SVDB
MITDB_PATH = "mitdb/"
SVDB_PATH = "svdb/"

# 🔹 Docelowa częstotliwość próbkowania
TARGET_FS = 360
SEGMENT_LENGTH = 200  # Długość segmentu w próbkach (QRS w środku)

# 🔹 Mapowanie etykiet
LABEL_MAP = {'N': 0, 'V': 1, 'A': 2, 'S': 3, 'L': 4, 'R': 5}
LABEL_MAP_SVDB = {'N': 0, 'V': 1, 'S': 2}
LABEL_MAP_MITDB = {'N': 0, 'V': 1, 'A': 2, 'L': 3, 'R': 4}
LABEL_NAMES = list(LABEL_MAP.keys())  # Kolejność klas
NUM_CLASSES = len(LABEL_MAP)





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

def filter_ecg(signal, fs):
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



def plot_confusion_matrix(y_true, y_pred, labels):
    """ Rysuje macierz błędów """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()






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





def balance_classes(X, y, class_to_reduce=0, reduction_factor=0.5):
    """🔄 Redukcja liczby segmentów klasy `class_to_reduce`."""
    idx_class = np.where(y == class_to_reduce)[0]  # Znajdź indeksy klasy "N"
    num_to_remove = int(len(idx_class) * reduction_factor)  # Określ liczbę do usunięcia

    idx_remove = np.random.choice(idx_class, num_to_remove, replace=False)  # Wylosuj do usunięcia
    idx_keep = np.setdiff1d(np.arange(len(y)), idx_remove)  # Indeksy, które zostawiamy

    return X[idx_keep], y[idx_keep]




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
                segment_len = len(segment)

                if segment_len < SEGMENT_LENGTH:
                    pad_left = (SEGMENT_LENGTH - segment_len) // 2
                    pad_right = SEGMENT_LENGTH - segment_len - pad_left
                    segment = np.pad(segment, (pad_left, pad_right), mode='edge')

                if len(segment) == SEGMENT_LENGTH:
                    signals.append(segment)
                    labels.append(LABEL_MAP[annotation.symbol[i]])

    return np.array(signals), np.array(labels)


def balance_classes_oversampling(X, y):
    """🔄 Oversampling klas mniejszościowych do liczby próbek klasy dominującej."""
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(len(X), -1), y)
    return X_resampled.reshape(len(X_resampled), SEGMENT_LENGTH), y_resampled


def build_cnn_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Masking(mask_value=0, input_shape=(SEGMENT_LENGTH, 1)),
        layers.Conv1D(128, kernel_size=15, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(256, kernel_size=9, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(512, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(128, return_sequences=False),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def train_model():
    print("Czy TensorFlow widzi GPU?", tf.config.list_physical_devices('GPU'))

    # Wczytanie rekordów MITDB i SVDB
    record_ids = sorted([f.split('.')[0] for f in os.listdir(MITDB_PATH) if f.endswith('.hea')])
    signals_mitdb, labels_mitdb = load_ecg_data(MITDB_PATH, record_ids)

    record_ids_svdb = sorted([f.split('.')[0] for f in os.listdir(SVDB_PATH) if f.endswith('.hea')])
    signals_svdb, labels_svdb = load_ecg_data(SVDB_PATH, record_ids_svdb)

    # Połączenie zbiorów
    X, y = np.concatenate((signals_mitdb, signals_svdb)), np.concatenate((labels_mitdb, labels_svdb))


    X, y = balance_classes(X,y,0,0.7)

    # Oversampling zamiast redukcji klasy dominującej
    X, y = balance_classes_oversampling(X, y)

    # Normalizacja
    X = (X - np.mean(X)) / np.std(X)

    # Podział na zbiory treningowe, walidacyjne i testowe
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16 , callbacks=[early_stopping, reduce_lr])

    # Ewaluacja modelu
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 🔹 Statystyki
    report = classification_report(y_true, y_pred_classes, target_names=LABEL_NAMES)
    print("\n📊 Statystyki modelu:\n", report)

    # 🔹 Macierz pomyłek
    plot_confusion_matrix(y_true, y_pred_classes, LABEL_NAMES)

    # Zapis modelu
    model.save("ecg_classifier.h5")

if __name__ == "__main__":
    train_model()

