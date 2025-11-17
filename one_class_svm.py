# one_class_svm.py
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


def one_class_svm_detection():
    print("=== 一类支持向量机异常检测 ===")

    # 加载数据
    data = pd.read_csv('creditcard.csv')

    # 准备特征
    features = ['V%d' % i for i in range(1, 29)] + ['Amount']
    X = data[features]
    y = data['Class']

    scaler = StandardScaler()

    X_normal = X[y == 0]
    X_fraud = X[y == 1]

    indices = np.arange(len(X_normal))
    train_size = int(np.floor(0.8 * len(X_normal)))
    train_indices = np.random.choice(len(X_normal), train_size, replace=False)
    test_indices = np.setdiff1d(indices, train_indices)

    train_data_raw = X_normal.iloc[train_indices]
    test_normal_raw = X_normal.iloc[test_indices]
    fraud_data_raw = X_fraud

    train_data = scaler.fit_transform(train_data_raw)
    test_normal = scaler.transform(test_normal_raw)
    fraud_data = scaler.transform(fraud_data_raw)

    # 训练OneClassSVM
    oc_svm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.04  # 异常值的上限比例
    )

    # 只用正常数据训练
    oc_svm.fit(train_data)

    X_test = np.vstack([test_normal, fraud_data])
    y_test = np.array([0] * len(test_normal) + [1] * len(fraud_data))
    y_pred = oc_svm.predict(X_test)
    y_pred_binary = [1 if x == -1 else 0 for x in y_pred]

    print("混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred_binary, labels=[1, 0])
    print(cm.T)

    decision_scores = oc_svm.decision_function(X_test)
    tp = cm[0, 0]
    fn = cm[0, 1]
    fp = cm[1, 0]
    tn = cm[1, 1]
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    g_mean = np.sqrt(recall * specificity)
    pr_auc = average_precision_score(y_test, -decision_scores)
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"AUPRC: {pr_auc:.4f}")
    print(f"G-Mean: {g_mean:.4f}")

    # 保存结果
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred_binary,
        'Anomaly_Score': -decision_scores
    })
    results.to_csv('one_class_svm_results.csv', index=False)

    return results


if __name__ == "__main__":
    one_class_svm_detection()