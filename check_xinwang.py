import pandas as pd
import numpy as np

# 加载xinwang数据集
df = pd.read_csv('data/xinwang.csv')

print('Xinwang数据集信息:')
print(f'总样本数: {len(df)}')
print(f'特征数: {len(df.columns)-1}')
print(f'\n目标变量分布:')
print(df['target'].value_counts())
print(f'\n目标变量比例:')
print(df['target'].value_counts(normalize=True))
print(f'\n类别不平衡比率: {df["target"].value_counts()[0] / df["target"].value_counts()[1]:.2f}:1')
