# autoencoder.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
import warnings

warnings.filterwarnings('ignore')


def autoencoder_detection():
    print("=== 自编码器异常检测 ===")

    # 加载数据
    data = pd.read_csv('creditcard.csv')

    # 准备特征
    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分离正常和欺诈交易
    normal_data = X_scaled[y == 0]
    fraud_data = X_scaled[y == 1]

    # 使用正常交易训练自编码器
    train_size = int(0.8 * len(normal_data))
    train_data = normal_data[:train_size]
    test_normal = normal_data[train_size:]

    input_dim = X_scaled.shape[1]
    encoding_dim = 14

    class AE(nn.Module):
        def __init__(self, input_dim, encoding_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, encoding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(encoding_dim * 2, encoding_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, encoding_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(encoding_dim * 2, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat

    model = AE(input_dim, encoding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    data_tensor = torch.tensor(train_data, dtype=torch.float32)
    n = data_tensor.shape[0]
    val_size = int(0.1 * n)
    train_tensor = data_tensor[:-val_size] if val_size > 0 else data_tensor
    val_tensor = data_tensor[-val_size:] if val_size > 0 else None

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

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
                _ = criterion(model(val_tensor), val_tensor)

    model.eval()
    with torch.no_grad():
        train_recon = []
        for i in range(0, len(train_data), 256):
            batch = torch.tensor(train_data[i:i+256], dtype=torch.float32)
            out = model(batch).cpu().numpy()
            train_recon.append(out)
        train_recon = np.vstack(train_recon)
    train_mse = np.mean(np.power(train_data - train_recon, 2), axis=1)
    threshold = np.percentile(train_mse, 95)

    X_test = np.vstack([test_normal, fraud_data])
    y_test = np.array([0] * len(test_normal) + [1] * len(fraud_data))

    with torch.no_grad():
        reconstructions = []
        for i in range(0, len(X_test), 256):
            batch = torch.tensor(X_test[i:i+256], dtype=torch.float32)
            out = model(batch).cpu().numpy()
            reconstructions.append(out)
        reconstructions = np.vstack(reconstructions)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    y_pred_binary = [1 if error > threshold else 0 for error in mse]

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
    pr_auc = average_precision_score(y_test, mse)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    # 保存结果
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_binary,
        'Reconstruction_Error': mse,
        'Anomaly_Score': mse,
        'Threshold': threshold
    })
    results.to_csv('autoencoder_results.csv', index=False)

    return results


if __name__ == "__main__":
    autoencoder_detection()