"""
消融实验分析器 - Ablation Study Analyzer
记录和可视化Top-k知识蒸馏中各参数的消融实验结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# 设置matplotlib为非交互式模式和字体
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')
plt.style.use('default')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_palette("husl")

class AblationStudyAnalyzer:
    """消融实验分析器"""
    
    def __init__(self):
        self.ablation_results = []
        self.experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def record_experiment_result(self, dataset_name, k, temperature, alpha, max_depth, accuracy, f1_score, precision, recall):
        """记录每次实验的结果"""
        result = {
            'dataset': dataset_name,
            'k': k,
            'temperature': temperature,
            'alpha': alpha,
            'max_depth': max_depth,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'timestamp': datetime.now().isoformat()
        }
        self.ablation_results.append(result)
        

        
    def create_ablation_visualizations(self):
        """创建消融实验可视化图 - TopK和决策树深度图（避免重复生成）"""
        if not self.ablation_results:
            print("❌ No ablation results to visualize")
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        # 数据集颜色映射 - 使用柔和好看的配色（与SHAP图一致）
        datasets = df['dataset'].unique()
        colors = ['#7BB3F0', '#DDA0DD', '#FFB366']  # 柔和蓝色、柔和紫色、柔和橙色
        dataset_colors = dict(zip(datasets, colors[:len(datasets)]))
        
        saved_plots = []
        
        # 1. Top-k特征数量分析 (如果数据中有k列且还没生成过)
        if 'k' in df.columns:
            topk_plot_path = f'results/topk_ablation_visualization_{self.experiment_timestamp}.png'
            # 检查是否已存在，避免重复生成
            import os
            if not os.path.exists(topk_plot_path):
                fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
                self._plot_topk_ablation(df, ax1, dataset_colors)
                plt.tight_layout()
                plt.savefig(topk_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_plots.append(topk_plot_path)
                print(f"✅ Top-k ablation plot saved: {topk_plot_path}")
            else:
                print(f"📋 Top-k ablation plot already exists: {topk_plot_path}")
        
        # 2. 决策树深度分析 (如果数据中有max_depth列且还没生成过)
        if 'max_depth' in df.columns:
            depth_plot_path = f'results/depth_ablation_visualization_{self.experiment_timestamp}.png'
            import os
            if not os.path.exists(depth_plot_path):
                fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
                self._plot_depth_ablation(df, ax2, dataset_colors)
                plt.tight_layout()
                plt.savefig(depth_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_plots.append(depth_plot_path)
                print(f"✅ Depth ablation plot saved: {depth_plot_path}")
            else:
                print(f"📋 Depth ablation plot already exists: {depth_plot_path}")
        
        print(f"✅ Ablation visualizations completed")
        return saved_plots
        
    def _plot_topk_ablation(self, df, ax, dataset_colors):
        """绘制Top-k特征数量的消融分析 - 曲线上每个点都是该k值的最高准确率"""
        max_points = []  # 存储每个数据集的整体最高点
        
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            
            # 按k值分组，取每个k值的最高准确率（而不是平均值）
            k_max_grouped = dataset_data.groupby('k')['accuracy'].max().reset_index()
            
            # 检查是否有数据
            if k_max_grouped.empty:
                print(f"⚠️  Warning: No k data found for dataset {dataset}")
                continue
            
            # 绘制曲线（使用每个k值的最高准确率）
            ax.plot(k_max_grouped['k'], k_max_grouped['accuracy'], 
                   label=dataset.upper(), marker='o', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
            
            # 找到整体最高点（在k_max_grouped中找最高的）
            max_idx = k_max_grouped['accuracy'].idxmax()
            max_k = k_max_grouped.loc[max_idx, 'k']
            max_acc = k_max_grouped.loc[max_idx, 'accuracy']
            max_points.append((max_k, max_acc, dataset))
            
            # 标记整体最高点（现在一定在曲线上）
            ax.scatter(max_k, max_acc, color=dataset_colors[dataset], 
                      s=120, marker='*', edgecolors='black', linewidth=1.5, zorder=5)
            
            # 添加垂直虚线从x轴到最高点
            ax.axvline(x=max_k, color=dataset_colors[dataset], 
                      linestyle='--', alpha=0.7, linewidth=1.5)
            
            # 添加最高点标注，根据k值智能调整偏移量避免重叠
            # 使用k值来决定标注位置，避免三个数据集的标注重叠在一起
            if max_k < 15:  # k值较小
                offset_x, offset_y = 12, 18  # 右上
            elif max_k < 30:  # k值中等
                offset_x, offset_y = 12, 0   # 右侧
            else:  # k值较大
                offset_x, offset_y = 12, -18  # 右下
                
            # 显示k值和准确率（显示4位小数）
            ax.annotate(f'k={max_k}\n{max_acc:.4f}', 
                       xy=(max_k, max_acc), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=10, color=dataset_colors[dataset],
                       fontweight='bold', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, 
                               edgecolor=dataset_colors[dataset], linewidth=1.2))
                               
            print(f"📊 {dataset.upper()} - 整体最优: k={max_k}, accuracy={max_acc:.4f} (在曲线上)")
                       
        ax.set_xlabel('Number of Top-k Features', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0.6, 1.0)  # 设置y轴范围从0.6到1.0
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')  # 图例放右上角
        
        # 设置x轴间隔为5，过滤掉None值
        k_values = sorted([k for k in df['k'].unique() if k is not None])
        if k_values:  # 如果有有效的k值
            ax.set_xticks([k for k in k_values if k % 5 == 0])  # x轴间隔为5
        else:
            # 如果没有k值，使用默认的x轴刻度
            ax.set_xticks(sorted(df['k'].dropna().unique()) if 'k' in df.columns else [])
        
    def _plot_temperature_ablation(self, df, ax, dataset_colors):
        """绘制温度参数的消融分析"""
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按温度分组，计算平均准确率
            temp_grouped = dataset_data.groupby('temperature')['accuracy'].mean().reset_index()
            
            ax.plot(temp_grouped['temperature'], temp_grouped['accuracy'],
                   label=dataset.upper(), marker='s', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
                       
        ax.set_xlabel('Temperature Parameter (T)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1.0)  # 设置y轴范围到1.0
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')  # 图例放右上角
        ax.set_xticks(sorted(df['temperature'].unique()))
        
    def _plot_alpha_ablation(self, df, ax, dataset_colors):
        """绘制加权参数α的消融分析 - 标记最高点版本"""
        max_points = []  # 存储每个数据集的最高点
        
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按α值分组，计算平均准确率
            alpha_grouped = dataset_data.groupby('alpha')['accuracy'].mean().reset_index()
            
            # 检查是否有数据
            if alpha_grouped.empty:
                print(f"⚠️  Warning: No alpha data found for dataset {dataset}")
                continue
            
            # 绘制曲线
            ax.plot(alpha_grouped['alpha'], alpha_grouped['accuracy'],
                   label=dataset.upper(), marker='^', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
            
            # 找到最高点
            max_idx = alpha_grouped['accuracy'].idxmax()
            max_alpha = alpha_grouped.loc[max_idx, 'alpha']
            max_acc = alpha_grouped.loc[max_idx, 'accuracy']
            max_points.append((max_alpha, max_acc, dataset))
            
            # 标记最高点
            ax.scatter(max_alpha, max_acc, color=dataset_colors[dataset], 
                      s=100, marker='*', edgecolors='black', linewidth=1, zorder=5)
            
            # 添加最高点标注，使用不同的偏移量和背景框避免重叠
            if dataset == 'uci':
                offset_x, offset_y = 5, 15  # UCI稍微高一点
            elif dataset == 'australian':
                offset_x, offset_y = 5, -15  # Australian稍微低一点
            else:  # german
                offset_x, offset_y = 5, 10  # German居中
                
            ax.annotate(f'{max_acc:.3f}', 
                       xy=(max_alpha, max_acc), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=9, color=dataset_colors[dataset],
                       fontweight='bold', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=dataset_colors[dataset]))
                       
        ax.set_xlabel('Weight Parameter (α)', fontsize=12, fontfamily='sans-serif')
        ax.set_ylabel('Accuracy', fontsize=12, fontfamily='sans-serif')
        ax.set_ylim(0.6, 1.0)  # 设置y轴范围从0.6到1.0
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)  # 图例放右上角
        if 'alpha' in df.columns:
            ax.set_xticks(sorted(df['alpha'].unique()))
        
    def _plot_depth_ablation(self, df, ax, dataset_colors):
        """绘制决策树深度的消融分析 - 曲线上每个点都是该depth值的最高准确率"""
        max_points = []  # 存储每个数据集的整体最高点
        
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            # 按深度分组，取每个depth值的最高准确率（而不是平均值）
            depth_max_grouped = dataset_data.groupby('max_depth')['accuracy'].max().reset_index()
            
            # 检查是否有数据
            if depth_max_grouped.empty:
                print(f"⚠️  Warning: No depth data found for dataset {dataset}")
                continue
            
            # 绘制曲线（使用每个depth值的最高准确率）
            ax.plot(depth_max_grouped['max_depth'], depth_max_grouped['accuracy'],
                   label=dataset.upper(), marker='d', linewidth=2, markersize=6,
                   color=dataset_colors[dataset])
            
            # 找到整体最高点（在depth_max_grouped中找最高的）
            max_idx = depth_max_grouped['accuracy'].idxmax()
            max_depth = depth_max_grouped.loc[max_idx, 'max_depth']
            max_acc = depth_max_grouped.loc[max_idx, 'accuracy']
            max_points.append((max_depth, max_acc, dataset))
            
            # 标记整体最高点（现在一定在曲线上）
            ax.scatter(max_depth, max_acc, color=dataset_colors[dataset], 
                      s=100, marker='*', edgecolors='black', linewidth=1, zorder=5)
            
            # 添加最高点标注，根据depth值智能调整偏移量避免重叠
            # 使用max_depth值来决定标注位置，避免三个数据集的标注重叠在一起
            if max_depth <= 5:  # depth较小
                offset_x, offset_y = 8, 18  # 右上
            elif max_depth <= 6:  # depth中等
                offset_x, offset_y = 8, 0   # 右侧
            else:  # depth较大
                offset_x, offset_y = 8, -18  # 右下
                
            ax.annotate(f'depth={int(max_depth)}\n{max_acc:.4f}', 
                       xy=(max_depth, max_acc), 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=10, color=dataset_colors[dataset],
                       fontweight='bold', ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.95, edgecolor=dataset_colors[dataset], linewidth=1.2))
            
            print(f"📊 {dataset.upper()} - 整体最优depth: {int(max_depth)}, accuracy={max_acc:.4f} (在曲线上)")
                       
        ax.set_xlabel('Decision Tree Max Depth', fontsize=12, fontfamily='sans-serif')
        ax.set_ylabel('Accuracy', fontsize=12, fontfamily='sans-serif')
        ax.set_ylim(0.6, 1.0)  # 设置y轴范围从0.6到1.0
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)  # 图例放右上角
        if 'max_depth' in df.columns:
            ax.set_xticks(sorted(df['max_depth'].unique()))
        
    def load_and_visualize_existing_data(self, data_path):
        """从已有数据文件加载数据"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.ablation_results = json.load(f)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            self.ablation_results = df.to_dict('records')
        else:
            raise ValueError("Data file must be JSON or CSV format")
            
        print(f"✅ Loaded ablation data from {data_path}")
        return []
        
    def generate_summary_report(self):
        """生成消融实验总结报告"""
        if not self.ablation_results:
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        report = []
        report.append("=" * 80)
        report.append("ABLATION STUDY SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Experiment Timestamp: {self.experiment_timestamp}")
        report.append(f"Total Experiments: {len(self.ablation_results)}")
        report.append(f"Datasets: {', '.join(df['dataset'].unique())}")
        report.append("")
        
        # 最佳配置分析
        for dataset in df['dataset'].unique():
            dataset_data = df[df['dataset'] == dataset]
            best_idx = dataset_data['accuracy'].idxmax()
            best_config = dataset_data.loc[best_idx]
            
            report.append(f"📊 {dataset.upper()} Dataset Best Configuration:")
            report.append(f"   • Accuracy: {best_config['accuracy']:.4f}")
            report.append(f"   • Top-k: {best_config['k']}")
            report.append(f"   • Temperature: {best_config['temperature']}")
            report.append(f"   • Alpha: {best_config['alpha']}")
            report.append(f"   • Max Depth: {best_config['max_depth']}")
            report.append("")
            
        # 参数影响分析
        report.append("🔍 Parameter Impact Analysis:")
        for param in ['k', 'temperature', 'alpha', 'max_depth']:
            correlation = df.groupby(param)['accuracy'].mean().corr(df.groupby(param).size())
            report.append(f"   • {param.upper()}: {correlation:.3f} correlation with accuracy")
            
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = f'results/ablation_study_report_{self.experiment_timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # 生成Excel报告
        excel_path = f'results/ablation_study_results_{self.experiment_timestamp}.xlsx'
        df.to_excel(excel_path, index=False)
        
        print(f"✅ Ablation study report saved: {report_path}")
        print(f"✅ Ablation study Excel saved: {excel_path}")
        print("\n" + report_text)
        
        return report_path





    def _plot_temperature_ablation(self, df, ax, dataset_colors):
        """绘制温度参数消融分析 - 无标题版本"""
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            temp_accuracy = dataset_df.groupby('temperature')['accuracy'].mean().reset_index()
            
            ax.plot(temp_accuracy['temperature'], temp_accuracy['accuracy'], 
                   marker='s', linewidth=2, markersize=6, 
                   color=dataset_colors[dataset], label=dataset.upper())
        
        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_ylim(0, 1.0)  # 设置y轴范围到1.0
        ax.legend(loc='upper right')  # 图例放右上角
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(df['temperature'].unique()))

    def save_ablation_data(self, prefix='ablation_study'):
        """保存消融实验数据 - 支持自定义前缀"""
        if not self.ablation_results:
            print("❌ No ablation data to save")
            return
        
        df = pd.DataFrame(self.ablation_results)
        
        # 保存CSV
        csv_path = f'results/{prefix}_{self.experiment_timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        # 保存JSON  
        json_path = f'results/{prefix}_{self.experiment_timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.ablation_results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {prefix} data saved: {csv_path}")
        print(f"✅ {prefix} data saved: {json_path}")

    def generate_summary_report(self, prefix='ablation_study'):
        """生成消融实验总结报告 - 支持自定义前缀"""
        if not self.ablation_results:
            print("❌ No ablation data to generate report")
            return None
            
        df = pd.DataFrame(self.ablation_results)
        
        report = []
        report.append("="*80)
        if 'topk' in prefix:
            report.append("Top-k Knowledge Distillation Ablation Study Report")
        else:
            report.append("All-Feature Knowledge Distillation Ablation Study Report")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total experiments: {len(df)}")
        report.append("")
        
        # 数据集统计
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            best_result = dataset_df.loc[dataset_df['accuracy'].idxmax()]
            
            report.append(f"📊 {dataset.upper()} Dataset:")
            report.append(f"   Best Accuracy: {best_result['accuracy']:.4f}")
            report.append(f"   Best F1-Score: {best_result['f1_score']:.4f}")
            
            if 'k' in best_result and best_result['k'] is not None:
                report.append(f"   Optimal k: {best_result['k']}")
            report.append(f"   Optimal α: {best_result['alpha']}")
            report.append(f"   Optimal Temperature: {best_result['temperature']}")
            report.append(f"   Optimal Max Depth: {best_result['max_depth']}")
            report.append("")
            
        report_text = "\n".join(report)
        
        # 保存报告
        report_path = f'results/{prefix}_report_{self.experiment_timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # 生成Excel报告
        excel_path = f'results/{prefix}_results_{self.experiment_timestamp}.xlsx'
        df.to_excel(excel_path, index=False)
        
        print(f"✅ {prefix} report saved: {report_path}")
        print(f"✅ {prefix} Excel saved: {excel_path}")
        print("\n" + report_text)
        
        return report_path

# 全局消融实验分析器实例
ablation_analyzer = AblationStudyAnalyzer()