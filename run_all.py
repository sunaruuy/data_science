# run_all.py
import subprocess
import sys


def run_all_methods():
    """运行所有异常检测方法"""

    methods = [
        'isolation_forest.py',
        'local_outlier_factor.py',
        'one_class_svm.py',
        'autoencoder.py',
        'lstm.py',
        'dbscan.py'
    ]

    print("开始运行所有异常检测方法...")

    for method in methods:
        print(f"\n{'=' * 50}")
        print(f"运行: {method}")
        print('=' * 50)

        try:
            subprocess.run([sys.executable, method], check=True)
        except subprocess.CalledProcessError as e:
            print(f"运行 {method} 时出错: {e}")
        except FileNotFoundError:
            print(f"找不到文件: {method}")

    # 最后比较所有方法
    print(f"\n{'=' * 50}")
    print("比较所有方法结果")
    print('=' * 50)
    try:
        subprocess.run([sys.executable, 'compare_methods.py'], check=True)
    except:
        print("比较方法时出错")


if __name__ == "__main__":
    run_all_methods()