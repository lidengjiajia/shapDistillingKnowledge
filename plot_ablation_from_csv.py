"""
从CSV文件生成消融实验图像
直接读取保存的消融实验数据，生成高质量的可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from datetime import datetime

# 设置matplotlib为非交互式模式
matplotlib.use('Agg')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_topk_ablation(csv_path, output_path=None):
    """
    从CSV生成Top-k消融实验图
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    print(f"📊 Loading Top-k ablation data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 排除xinwang数据集
    df = df[df['dataset'] != 'xinwang']
    
    # 数据集颜色映射
    datasets = df['dataset'].unique()
    colors = ['#7BB3F0', '#DDA0DD', '#FFB366', '#90EE90']
    dataset_colors = dict(zip(datasets, colors[:len(datasets)]))
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    
    # 收集所有最高点信息，用于智能标注
    max_points_info = []
    
    # 为每个数据集绘制曲线
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        
        # 按k值排序
        dataset_data = dataset_data.sort_values('k')
        
        # 绘制曲线 - 使用更细的线条避免干扰
        ax.plot(dataset_data['k'], dataset_data['accuracy'], 
               label=dataset.upper(), marker='o', linewidth=2, markersize=6,
               color=dataset_colors[dataset], alpha=0.85, markeredgewidth=0.5,
               markeredgecolor='white')
        
        # 找到最高点
        max_idx = dataset_data['accuracy'].idxmax()
        max_k = dataset_data.loc[max_idx, 'k']
        max_acc = dataset_data.loc[max_idx, 'accuracy']
        
        max_points_info.append({
            'dataset': dataset,
            'k': max_k,
            'acc': max_acc,
            'color': dataset_colors[dataset]
        })
        
        print(f"   {dataset.upper()}: Best k={int(max_k)}, Accuracy={max_acc:.4f}")
    
    # 按k值排序，智能调整标注位置避免重叠
    max_points_info.sort(key=lambda x: x['k'])
    
    for i, point_info in enumerate(max_points_info):
        max_k = point_info['k']
        max_acc = point_info['acc']
        dataset_color = point_info['color']
        
        # 标记最高点 - 使用更大的星号
        ax.scatter(max_k, max_acc, color=dataset_color, 
                  s=250, marker='*', edgecolors='#2C3E50', linewidth=2.5, zorder=10)
        
        # 添加细虚线到x轴
        ax.plot([max_k, max_k], [0.65, max_acc], 
               color=dataset_color, linestyle=':', alpha=0.4, linewidth=1.2, zorder=1)
        
        # 智能标注位置 - 保持在第一象限内，大幅拉开距离
        # 根据实际数据：german(k=12), australian(k=18), uci(k=21)
        if i == 0:  # 第一个点（k=12, german, acc=0.74）
            offset_x, offset_y = 25, -55  # 向右下大幅偏移
            ha = 'left'
        elif i == 1:  # 第二个点（k=18, australian, acc=0.848）
            offset_x, offset_y = 20, 45  # 向右上偏移，避免遮挡
            ha = 'left'
        else:  # 第三个点（k=21, uci黄色, acc=0.82）
            offset_x, offset_y = 30, 50  # 向右上角偏移，避免遮挡黄色曲线
            ha = 'left'
        
        ax.annotate(f'k={int(max_k)}\n{max_acc:.4f}', 
                   xy=(max_k, max_acc), 
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=10, color=dataset_color,
                   fontweight='bold', ha=ha,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, 
                           edgecolor=dataset_color, linewidth=1.8),
                   arrowprops=dict(arrowstyle='->', color=dataset_color, lw=1.2, alpha=0.6),
                   zorder=11)
    
    # 设置x轴范围 - 只显示有数据的范围
    all_k_values = sorted(df['k'].unique())
    min_k = min(all_k_values)
    max_k = max(all_k_values)
    k_range = max_k - min_k
    
    # 设置x轴刻度 - 智能间隔
    if k_range <= 15:
        interval = 1
    elif k_range <= 30:
        interval = 2
    elif k_range <= 50:
        interval = 5
    else:
        interval = 10
    
    tick_values = [k for k in all_k_values if k % interval == 0 or k == min_k or k == max_k]
    ax.set_xticks(tick_values)
    ax.set_xlim(min_k - 0.5, max_k + 0.5)  # 只显示数据范围，留少量边距
    
    # 设置y轴
    ax.set_ylim(0.65, 1.0)
    ax.set_xlabel('Number of Top-k Features', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_title('Top-k Feature Ablation Study', fontsize=16, fontweight='bold', 
                color='#1A252F', pad=20)
    
    # 美化网格和图例
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, 
             edgecolor='#2C3E50', fancybox=True, shadow=True)
    
    # 设置背景
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # 边框美化
    for spine in ax.spines.values():
        spine.set_edgecolor('#2C3E50')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/topk_ablation_plot_{timestamp}.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Top-k ablation plot saved: {output_path}")
    return output_path

def plot_depth_ablation(csv_path, output_path=None):
    """
    从CSV生成决策树深度消融实验图
    
    Args:
        csv_path: CSV文件路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    print(f"📊 Loading Depth ablation data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 排除xinwang数据集
    df = df[df['dataset'] != 'xinwang']
    
    # 数据集颜色映射
    datasets = df['dataset'].unique()
    colors = ['#7BB3F0', '#DDA0DD', '#FFB366', '#90EE90']
    dataset_colors = dict(zip(datasets, colors[:len(datasets)]))
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=150)
    
    # 收集所有最高点信息，用于智能标注
    max_points_info = []
    
    # 为每个数据集绘制曲线
    for dataset in datasets:
        dataset_data = df[df['dataset'] == dataset]
        
        # 按深度值排序
        dataset_data = dataset_data.sort_values('max_depth')
        
        # 绘制曲线 - 使用更细的线条避免干扰
        ax.plot(dataset_data['max_depth'], dataset_data['accuracy'], 
               label=dataset.upper(), marker='d', linewidth=2, markersize=7,
               color=dataset_colors[dataset], alpha=0.85, markeredgewidth=0.5,
               markeredgecolor='white')
        
        # 找到最高点
        max_idx = dataset_data['accuracy'].idxmax()
        max_depth = dataset_data.loc[max_idx, 'max_depth']
        max_acc = dataset_data.loc[max_idx, 'accuracy']
        
        max_points_info.append({
            'dataset': dataset,
            'depth': max_depth,
            'acc': max_acc,
            'color': dataset_colors[dataset]
        })
        
        print(f"   {dataset.upper()}: Best depth={int(max_depth)}, Accuracy={max_acc:.4f}")
    
    # 按深度值排序，智能调整标注位置避免重叠
    max_points_info.sort(key=lambda x: x['depth'])
    
    for i, point_info in enumerate(max_points_info):
        max_depth = point_info['depth']
        max_acc = point_info['acc']
        dataset_color = point_info['color']
        
        # 标记最高点 - 使用更大的星号
        ax.scatter(max_depth, max_acc, color=dataset_color, 
                  s=250, marker='*', edgecolors='#2C3E50', linewidth=2.5, zorder=10)
        
        # 添加细虚线到x轴
        ax.plot([max_depth, max_depth], [0.65, max_acc], 
               color=dataset_color, linestyle=':', alpha=0.4, linewidth=1.2, zorder=1)
        
        # 智能标注位置 - 保持在第一象限内，大幅拉开距离
        # 根据实际数据：uci(depth=4, acc=0.82), german(depth=6, acc=0.74), australian(depth=7, acc=0.848)
        if i == 0:  # depth=4 (UCI, acc=0.82)
            offset_x, offset_y = 25, 45  # 向右上大幅偏移，保持在第一象限
            ha = 'left'
        elif i == 1:  # depth=6 (German, acc=0.74)
            offset_x, offset_y = 25, -55  # 向右下大幅偏移
            ha = 'left'
        else:  # depth=7 (Australian, acc=0.848)
            offset_x, offset_y = 30, 40  # 向右上大幅偏移
            ha = 'left'
        
        ax.annotate(f'depth={int(max_depth)}\n{max_acc:.4f}', 
                   xy=(max_depth, max_acc), 
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=10, color=dataset_color,
                   fontweight='bold', ha=ha,
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, 
                           edgecolor=dataset_color, linewidth=1.8),
                   arrowprops=dict(arrowstyle='->', color=dataset_color, lw=1.2, alpha=0.6),
                   zorder=11)
    
    # 设置x轴范围 - 只显示有数据的范围
    all_depth_values = sorted(df['max_depth'].unique())
    min_depth = min(all_depth_values)
    max_depth = max(all_depth_values)
    
    ax.set_xticks(all_depth_values)  # 深度值通常较少，全部显示
    ax.set_xlim(min_depth - 0.3, max_depth + 0.3)  # 只显示数据范围，留少量边距
    
    # 设置y轴
    ax.set_ylim(0.65, 1.0)
    ax.set_xlabel('Decision Tree Max Depth', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_title('Decision Tree Depth Ablation Study', fontsize=16, fontweight='bold', 
                color='#1A252F', pad=20)
    
    # 美化网格和图例
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#7F8C8D')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95, 
             edgecolor='#2C3E50', fancybox=True, shadow=True)
    
    # 设置背景
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    # 边框美化
    for spine in ax.spines.values():
        spine.set_edgecolor('#2C3E50')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/depth_ablation_plot_{timestamp}.png'
    
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Depth ablation plot saved: {output_path}")
    return output_path

def find_latest_csv(pattern):
    """查找最新的CSV文件"""
    import glob
    files = glob.glob(f'results/{pattern}*.csv')
    if not files:
        return None
    return max(files, key=os.path.getmtime)

if __name__ == '__main__':
    print("="*80)
    print("📊 消融实验图像生成器")
    print("="*80)
    
    # 1. 生成Top-k图
    print("\n1️⃣  Generating Top-k Ablation Plot...")
    topk_csv = find_latest_csv('topk_ablation_data')
    if topk_csv:
        plot_topk_ablation(topk_csv)
    else:
        print("❌ No Top-k ablation data found!")
    
    # 2. 生成Depth图
    print("\n2️⃣  Generating Depth Ablation Plot...")
    depth_csv = find_latest_csv('depth_ablation_data')
    if depth_csv:
        plot_depth_ablation(depth_csv)
    else:
        print("❌ No Depth ablation data found!")
    
    print("\n" + "="*80)
    print("✅ 所有图像生成完成！")
    print("="*80)
