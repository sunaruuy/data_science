import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def dbscan_detection():
    print("=== DBSCAN聚类异常检测 ===")

    data = pd.read_csv('creditcard.csv')

    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sample_size = min(50000, len(X_scaled))
    if len(X_scaled) > sample_size:
        indices = np.random.choice(len(X_scaled), sample_size, replace=False)
        X_sampled = X_scaled[indices]
        y_sampled = y.iloc[indices]
    else:
        X_sampled = X_scaled
        y_sampled = y

    min_samples = 10

    X_eps = X_sampled

    nn_eps = NearestNeighbors(n_neighbors=min_samples, n_jobs=-1)
    nn_eps.fit(X_eps)
    distances_eps, _ = nn_eps.kneighbors(X_eps)
    k_distance_eps = distances_eps[:, -1]
    eps = np.percentile(k_distance_eps, 95)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(X_sampled)
    y_pred_binary = [1 if x == -1 else 0 for x in labels]

    print("混淆矩阵:")
    cm = confusion_matrix(y_sampled, y_pred_binary, labels=[1, 0])
    print(cm.T)

    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    precision = precision_score(y_sampled, y_pred_binary, zero_division=0)
    recall = recall_score(y_sampled, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall * specificity)
    pr_auc = average_precision_score(y_sampled, y_pred_binary)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    results = pd.DataFrame({
        'Actual': y_sampled,
        'Predicted': y_pred_binary,
        'Anomaly_Score': y_pred_binary
    })
    results.to_csv('dbscan_results.csv', index=False)

    return results


if __name__ == "__main__":
    dbscan_detection()