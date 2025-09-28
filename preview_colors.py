#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def create_color_preview():
    """创建配色预览图"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 旧配色（深色）
    old_colors = ['#2E86AB', '#A23B72', '#F18F01']
    old_names = ['German (深蓝)', 'Australian (深紫)', 'UCI (深橙)']
    
    # 新配色（柔和）
    new_colors = ['#7BB3F0', '#DDA0DD', '#FFB366'] 
    new_names = ['German (柔和蓝)', 'Australian (柔和紫)', 'UCI (柔和橙)']
    
    # 绘制旧配色
    y_pos = np.arange(len(old_colors))
    ax1.barh(y_pos, [1]*3, color=old_colors, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(old_names)
    ax1.set_title('旧配色 (太深)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.2)
    
    # 绘制新配色
    ax2.barh(y_pos, [1]*3, color=new_colors, alpha=0.9)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(new_names)
    ax2.set_title('新配色 (柔和)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig('results/color_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 配色对比图已��存: results/color_comparison.png")
    
    # 打印颜色值对比
    print("\n🎨 配色对比:")
    for i, (old, new, name) in enumerate(zip(old_colors, new_colors, ['German', 'Australian', 'UCI'])):
        print(f"   {name:10s}: {old} → {new}")

if __name__ == "__main__":
    create_color_preview()