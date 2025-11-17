import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

    class LSTMAE(nn.Module):
        def __init__(self, input_dim, hidden_dim, seq_len):
            super().__init__()
            self.seq_len = seq_len
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.to_out = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            _, (h_n, _) = self.encoder(x)
            z = h_n[-1]
            rep = z.unsqueeze(1).repeat(1, self.seq_len, 1)
            y, _ = self.decoder(rep)
            out = self.to_out(y)
            return out

    model = LSTMAE(input_dim, 64, seq_len)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    data_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    n = data_tensor.shape[0]
    val_size = int(0.1 * n)
    train_tensor = data_tensor[:-val_size] if val_size > 0 else data_tensor
    val_tensor = data_tensor[-val_size:] if val_size > 0 else None

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = None
    patience = 5
    wait = 0
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            bs = batch.shape[0]
            total_loss += loss.item() * bs
            total_count += bs
        epoch_loss = total_loss / max(total_count, 1)
        print(f"Epoch {epoch+1}: loss {epoch_loss:.6f}")
        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                val_recon = model(val_tensor)
                val_loss = criterion(val_recon, val_tensor).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

    train_eval_sequences = _create_sequences(train_block, seq_len)
    model.eval()
    with torch.no_grad():
        recon_train_batches = []
        for i in range(0, len(train_eval_sequences), 128):
            batch = torch.tensor(train_eval_sequences[i:i+128], dtype=torch.float32)
            out = model(batch).cpu().numpy()
            recon_train_batches.append(out)
        recon_train = np.vstack(recon_train_batches)
    seq_err_train = _sequence_errors(train_eval_sequences, recon_train)
    row_err_train = _row_errors_from_sequence_errors(seq_err_train, len(train_block), seq_len)
    threshold = np.percentile(row_err_train, 20)

    test_normal_block = scaler.transform(normal_data_raw[train_size:])
    fraud_block = scaler.transform(X_sampled_raw[y_sampled == 1])
    X_test_block = np.vstack([test_normal_block, fraud_block])
    y_test = np.array([0] * len(test_normal_block) + [1] * len(fraud_block))

    eval_sequences = _create_sequences(X_test_block, seq_len)
    with torch.no_grad():
        recon_batches = []
        for i in range(0, len(eval_sequences), 128):
            batch = torch.tensor(eval_sequences[i:i+128], dtype=torch.float32)
            out = model(batch).cpu().numpy()
            recon_batches.append(out)
        recon = np.vstack(recon_batches)
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