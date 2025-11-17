# autoencoder.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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

    # 构建自编码器
    input_dim = X_scaled.shape[1]
    encoding_dim = 14

    # 编码器
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim * 2, activation='relu')(input_layer)
    encoder = Dropout(0.1)(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)

    # 解码器
    decoder = Dense(encoding_dim * 2, activation='relu')(encoder)
    decoder = Dropout(0.1)(decoder)
    decoder = Dense(input_dim, activation='linear')(decoder)

    # 自编码器模型
    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # 编译模型
    autoencoder.compile(optimizer=Adam(learning_rate=0.001),
                        loss='mse')

    # 训练模型
    history = autoencoder.fit(
        train_data, train_data,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    train_recon = autoencoder.predict(train_data, batch_size=256)
    train_mse = np.mean(np.power(train_data - train_recon, 2), axis=1)
    threshold = np.percentile(train_mse, 95)

    X_test = np.vstack([test_normal, fraud_data])
    y_test = np.array([0] * len(test_normal) + [1] * len(fraud_data))

    reconstructions = autoencoder.predict(X_test, batch_size=256)
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