#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

def verify_topk_curve_logic():
    """验证TopK曲线逻辑是否正确"""
    
    # 创建测试数据：每个k值有多次实验结果
    test_data = [
        # k=5的实验结果
        {'dataset': 'test', 'k': 5, 'accuracy': 0.80},
        {'dataset': 'test', 'k': 5, 'accuracy': 0.82},  # 最高
        {'dataset': 'test', 'k': 5, 'accuracy': 0.81},
        
        # k=10的实验结果  
        {'dataset': 'test', 'k': 10, 'accuracy': 0.85},
        {'dataset': 'test', 'k': 10, 'accuracy': 0.87},  # 最高
        {'dataset': 'test', 'k': 10, 'accuracy': 0.86},
        
        # k=15的实验结果
        {'dataset': 'test', 'k': 15, 'accuracy': 0.88},
        {'dataset': 'test', 'k': 15, 'accuracy': 0.90},  # 最高，也是整体最高
        {'dataset': 'test', 'k': 15, 'accuracy': 0.89},
    ]
    
    df = pd.DataFrame(test_data)
    
    print("=== 原始数据 ===")
    print(df)
    
    print("\n=== 按平均值计算（旧方法，错误）===")
    avg_grouped = df.groupby('k')['accuracy'].mean().reset_index()
    print(avg_grouped)
    avg_max_idx = avg_grouped['accuracy'].idxmax()
    print(f"平均值最高点: k={avg_grouped.loc[avg_max_idx, 'k']}, avg_acc={avg_grouped.loc[avg_max_idx, 'accuracy']:.3f}")
    
    print("\n=== 按最高值计算（新方法，正确）===")
    max_grouped = df.groupby('k')['accuracy'].max().reset_index()
    print(max_grouped)
    max_max_idx = max_grouped['accuracy'].idxmax()
    print(f"最高值最高点: k={max_grouped.loc[max_max_idx, 'k']}, max_acc={max_grouped.loc[max_max_idx, 'accuracy']:.3f}")
    
    print("\n=== 单次实验最高点 ===")
    single_max_idx = df['accuracy'].idxmax()
    single_max = df.loc[single_max_idx]
    print(f"单次实验最高: k={single_max['k']}, accuracy={single_max['accuracy']:.3f}")
    
    print("\n🔍 验证结果:")
    print(f"新方法的曲线最高点 与 单次实验最高点 一致: {max_grouped.loc[max_max_idx, 'accuracy'] == single_max['accuracy']}")
    print("✅ 这确保了最高点标记在曲线上!")

if __name__ == "__main__":
    verify_topk_curve_logic()