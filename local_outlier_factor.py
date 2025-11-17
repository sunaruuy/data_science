# local_outlier_factor.py
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def lof_detection():
    print("=== 局部离群因子异常检测 ===")

    data = pd.read_csv('creditcard.csv')

    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_sampled = X_scaled
    y_sampled = y

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.0017,
        novelty=False,
        n_jobs=-1
    )

    labels = lof.fit_predict(X_sampled)
    y_pred_binary = [1 if x == -1 else 0 for x in labels]

    print("混淆矩阵:")
    cm = confusion_matrix(y_sampled, y_pred_binary, labels=[1, 0])
    print(cm.T)

    scores = -lof.negative_outlier_factor_
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    precision = precision_score(y_sampled, y_pred_binary, zero_division=0)
    recall = recall_score(y_sampled, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall * specificity)
    pr_auc = average_precision_score(y_sampled, scores)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    results = pd.DataFrame({
        'Actual': y_sampled,
        'Predicted': y_pred_binary,
        'Anomaly_Score': scores
    })
    results.to_csv('lof_results.csv', index=False)

    return results


if __name__ == "__main__":
    lof_detection()