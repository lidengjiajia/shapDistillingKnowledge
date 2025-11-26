"""
测试基线模型在xinwang数据集上的性能(验证类别不平衡处理)
"""
import sys
sys.path.append('.')

from baseline_models import BaselineModels
from data_preprocessing import DataPreprocessor
import numpy as np

print("="*80)
print("测试基线模型 - Xinwang数据集类别不平衡处理")
print("="*80)

# 加载数据
preprocessor = DataPreprocessor()
data = preprocessor.load_xinwang()

if data is None:
    print("❌ Failed to load xinwang data")
    sys.exit(1)

X_train, X_val, X_test, y_train, y_val, y_test = data

print(f"\n数据形状:")
print(f"  训练集: {X_train.shape}, 标签: {y_train.shape}")
print(f"  验证集: {X_val.shape}, 标签: {y_val.shape}")
print(f"  测试集: {X_test.shape}, 标签: {y_test.shape}")

print(f"\n类别分布:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
print(f"  训练集: {dict(zip(unique_train, counts_train))}")
unique_test, counts_test = np.unique(y_test, return_counts=True)
print(f"  测试集: {dict(zip(unique_test, counts_test))}")

# 训练一个逻辑回归模型作为测试
print("\n" + "="*80)
print("测试 Logistic Regression (L2) - 带类别权重")
print("="*80)

baseline = BaselineModels(random_state=42)
model, params = baseline.train_logistic(X_train, y_train, X_val, y_val, n_trials=20, penalty='l2')

# 评估
metrics = baseline._evaluate_model(model, X_test, y_test)

print(f"\n测试集性能:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1-Score:  {metrics['f1_score']:.4f}")

print("\n✅ 如果precision、recall和f1不再是0,说明修复成功!")
