# isolation_forest.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def isolation_forest_detection():
    print("=== 孤立森林异常检测 ===")

    # 加载数据
    data = pd.read_csv('creditcard.csv')

    # 准备特征
    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练孤立森林模型
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.0017,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_scaled)
    y_pred = iso_forest.predict(X_scaled)

    # 转换预测结果（孤立森林返回-1/1，我们需要转换为0/1）
    y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

    print("混淆矩阵:")
    cm = confusion_matrix(y, y_pred_binary, labels=[1, 0])
    print(cm.T)

    anomaly_scores = iso_forest.decision_function(X_scaled)
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    precision = precision_score(y, y_pred_binary, zero_division=0)
    recall = recall_score(y, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall * specificity)
    pr_auc = average_precision_score(y, -anomaly_scores)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    # 保存结果
    results = pd.DataFrame({
        'Actual': y,
        'Predicted': y_pred_binary,
        'Anomaly_Score': -anomaly_scores
    })
    results.to_csv('isolation_forest_results.csv', index=False)

    return results


if __name__ == "__main__":
    isolation_forest_detection()