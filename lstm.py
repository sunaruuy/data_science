import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')


def _create_sequences(X, seq_len):
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i + seq_len])
    return np.array(sequences)


def _sequence_errors(x_true, x_pred):
    return np.mean(np.power(x_true - x_pred, 2), axis=(1, 2))


def _row_errors_from_sequence_errors(seq_err, n_rows, seq_len):
    row_err = np.zeros(n_rows)
    counts = np.zeros(n_rows)
    for s_idx in range(len(seq_err)):
        start = s_idx
        end = s_idx + seq_len
        row_err[start:end] += seq_err[s_idx]
        counts[start:end] += 1
    counts[counts == 0] = 1
    return row_err / counts


def lstm_detection():
    print("=== LSTM自编码器异常检测 ===")

    data = pd.read_csv('creditcard.csv')

    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    scaler = StandardScaler()

    sample_size = len(X)
    X_sampled_raw = X.iloc[:sample_size]
    y_sampled = y.iloc[:sample_size]

    seq_len = 30
    input_dim = X_sampled_raw.shape[1]

    normal_data_raw = X_sampled_raw[y_sampled == 0]
    train_size = int(0.8 * len(normal_data_raw))
    train_block_raw = normal_data_raw[:train_size]

    scaler.fit(train_block_raw)
    train_block = scaler.transform(train_block_raw)
    train_sequences = _create_sequences(train_block, seq_len)

    inp = Input(shape=(seq_len, input_dim))
    enc = LSTM(64, activation='tanh', return_sequences=False)(inp)
    rep = RepeatVector(seq_len)(enc)
    dec = LSTM(64, activation='tanh', return_sequences=True)(rep)
    out = TimeDistributed(Dense(input_dim))(dec)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        train_sequences, train_sequences,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_split=0.1,
        callbacks=[es],
        verbose=1
    )

    train_eval_sequences = _create_sequences(train_block, seq_len)
    recon_train = model.predict(train_eval_sequences, batch_size=128, verbose=0)
    seq_err_train = _sequence_errors(train_eval_sequences, recon_train)
    row_err_train = _row_errors_from_sequence_errors(seq_err_train, len(train_block), seq_len)
    threshold = np.percentile(row_err_train, 20)

    test_normal_block = scaler.transform(normal_data_raw[train_size:])
    fraud_block = scaler.transform(X_sampled_raw[y_sampled == 1])
    X_test_block = np.vstack([test_normal_block, fraud_block])
    y_test = np.array([0] * len(test_normal_block) + [1] * len(fraud_block))

    eval_sequences = _create_sequences(X_test_block, seq_len)
    recon = model.predict(eval_sequences, batch_size=128, verbose=0)
    seq_err = _sequence_errors(eval_sequences, recon)
    row_err = _row_errors_from_sequence_errors(seq_err, len(X_test_block), seq_len)
    y_pred_binary = [1 if e > threshold else 0 for e in row_err]

    print("混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred_binary, labels=[1, 0])
    print(cm.T)

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall * specificity)
    pr_auc = average_precision_score(y_test, row_err)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_binary,
        'Reconstruction_Error': row_err,
        'Anomaly_Score': row_err,
        'Threshold': threshold
    })
    results.to_csv('lstm_results.csv', index=False)
    return results


if __name__ == "__main__":
    lstm_detection()