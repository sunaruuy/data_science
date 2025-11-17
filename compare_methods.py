# compare_methods.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.metrics import roc_curve, auc, precision_recall_curve

_candidate_fonts = ['PingFang SC', 'Hiragino Sans GB', 'Songti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for _f in _candidate_fonts:
    if _f in _available_fonts:
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False


def compare_methods():
    print("=== 方法比较 ===")

    methods = ['Isolation Forest', 'LOF', 'One-Class SVM', 'Autoencoder', 'LSTM', 'Transformer', 'DBSCAN']
    files = ['isolation_forest_results.csv', 'lof_results.csv', 'one_class_svm_results.csv', 'autoencoder_results.csv', 'lstm_results.csv', 'transformer_results.csv', 'dbscan_results.csv']

    data = []
    for method, file in zip(methods, files):
        try:
            df = pd.read_csv(file)
            score_col = 'Anomaly_Score' if 'Anomaly_Score' in df.columns else ('Reconstruction_Error' if 'Reconstruction_Error' in df.columns else None)
            if score_col is None:
                continue
            y_true = df['Actual'].values
            y_pred = df['Predicted'].values if 'Predicted' in df.columns else None
            scores = df[score_col].values
            data.append({'method': method, 'y_true': y_true, 'y_pred': y_pred, 'scores': scores})
        except:
            continue

    if len(data) == 0:
        print("未找到任何结果文件")
        return

    plt.figure(figsize=(8, 6))
    for item in data:
        precision, recall, _ = precision_recall_curve(item['y_true'], item['scores'])
        plt.plot(recall, precision, label=item['method'])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('PR曲线 - 核心评估图')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pr_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    for item in data:
        fpr, tpr, _ = roc_curve(item['y_true'], item['scores'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{item['method']} (AUC={roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线 - 与PR曲线对比')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    cols = 3
    rows = int(np.ceil(len(data) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = np.array(axes).reshape(rows, cols)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < len(data):
                item = data[idx]
                df_plot = pd.DataFrame({'score': item['scores'], 'label': np.where(item['y_true'] == 1, '欺诈', '正常')})
                sns.histplot(df_plot, x='score', hue='label', bins=50, stat='density', common_norm=False, ax=ax, element='step')
                ax.set_title(item['method'])
                ax.set_xlabel('异常分数')
                ax.set_ylabel('密度')
                ax.grid(True)
            else:
                ax.axis('off')
            idx += 1
    fig.suptitle('概率分布直方图 - 模型区分度')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig('score_distribution_hist.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('已生成: pr_curve_comparison.png, roc_curve_comparison.png, score_distribution_hist.png')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    images = [
        ('pr_curve_comparison.png', 'PR曲线 - 核心评估图'),
        ('roc_curve_comparison.png', 'ROC曲线 - 与PR曲线对比'),
        ('score_distribution_hist.png', '概率分布直方图 - 模型区分度')
    ]
    for ax, (path, title) in zip(axes.flatten(), images):
        try:
            img = plt.imread(path)
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
        except:
            ax.text(0.5, 0.5, f'缺少: {path}', ha='center', va='center')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig('methods_all_in_one.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('已合成总览图: methods_all_in_one.png')


if __name__ == "__main__":
    compare_methods()