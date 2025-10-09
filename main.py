"""
信用评分模型优化系统 - 主程序
Credit Scoring Model Optimization System - Main Program

基于SHAP特征重要性分析和知识蒸馏的信用评分模型优化系统
模块化架构，支持扩展参数组合和改进的用户体验
"""

import os
import warnings
import torch
import numpy as np
from tqdm import tqdm
import random

# ============================
# 全局随机种子设置 - 确保实验可重复
# ============================
def set_global_seed(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在导入任何模块之前设置随机种子
set_global_seed(42)

# 解决中文路径编码问题
import locale
import tempfile
import multiprocessing
try:
    locale.setlocale(locale.LC_ALL, 'C')
except:
    pass

# 设置临时目录为英文路径避免编码问题
temp_dir = "C:\\temp_ml"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
os.environ['TEMP'] = temp_dir
os.environ['TMP'] = temp_dir

# 导入自定义模块
from data_preprocessing import DataPreprocessor
from neural_models import train_all_teacher_models
from shap_analysis import SHAPAnalyzer
from distillation_module import KnowledgeDistillator
from result_manager import ResultManager
from ablation_analyzer import AblationStudyAnalyzer

warnings.filterwarnings('ignore')

def main():
    """主函数 - 运行完整的信用评分模型优化系统"""
    
    from multiprocessing import cpu_count
    
    print("="*80)
    print("🎯 信用评分模型优化系统 | Credit Scoring Model Optimization System")
    print("   基于SHAP特征重要性分析和知识蒸馏 | SHAP + Knowledge Distillation")
    print("   增强版 - 支持决策树深度参数和消融实验分析")
    print("="*80)
    print("🔧 并发配置:")
    print(f"   • CPU核心数: {cpu_count()}")
    print(f"   • 并发工作进程: {max(1, min(cpu_count() - 1, cpu_count()))} (保留1个核心给系统)")
    print("="*80)
    print("📊 实验参数配置:")
    print("   • Top-k特征: k=动态范围 (5到每个数据集的特征总数)")
    print("     - German: k=5到54 (50个值)")
    print("     - Australian: k=5到22 (18个值)")
    print("     - UCI: k=5到23 (19个值)")
    print("   • 加权比例参数: α=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0") 
    print("   • 温度参数: T=1,2,3,4,5") 
    print("   • 决策树深度: D=4,5,6,7,8")
    print("   • 基线模型: 固定参数（max_depth=5），无参数优化")
    print("   • 决策树学生模型知识蒸馏")
    print("="*80)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    try:
        # ========================
        # 1. 数据预处理阶段
        # ========================
        print(f"\n🔄 Phase 1: Data Preprocessing")
        print(f"   Loading and preprocessing datasets...")
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_all_datasets()
        
        print(f"   ✅ Data preprocessing completed")
        for dataset_name, data_dict in processed_data.items():
            print(f"     • {dataset_name.upper()}: {data_dict['X_train'].shape[0]} train, {data_dict['X_test'].shape[0]} test samples")
        
        # ========================
        # 2. 教师模型训练阶段
        # ========================
        teacher_models = train_all_teacher_models(processed_data)
        
        # ========================
        # 3. SHAP特征重要性分析
        # ========================
        print(f"\n🔍 Phase 3: SHAP Feature Importance Analysis")
        print(f"   Training decision trees for SHAP analysis...")
        
        shap_analyzer = SHAPAnalyzer(processed_data)
        
        # 先训练决策树模型用于SHAP分析
        shap_analyzer.train_decision_trees()
        
        # 计算SHAP值 - 使用动态k范围
        all_shap_results = {}
        for dataset_name in ['uci', 'german', 'australian']:
            # 每个数据集使用其特征总数作为k的最大值
            data_dict = processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            all_shap_results[dataset_name] = shap_analyzer.compute_shap_values(
                dataset_name, 
                top_k_range=(5, n_features)
            )
        
        # 创建组合SHAP可视化
        shap_viz_path = shap_analyzer.create_combined_shap_visualization(all_shap_results)
        
        print(f"   ✅ SHAP analysis completed")
        print(f"     • Combined visualization: {shap_viz_path}")
        
        # ========================
        # 初始化知识蒸馏器
        # ========================
        distillator = KnowledgeDistillator(teacher_models, processed_data, all_shap_results)
        
        # ========================
        # 4. 基础决策树训练（对比基准）
        # ========================
        print(f"\n🌳 Phase 4: Baseline Decision Tree Training")
        print(f"   Training baseline decision trees for comparison...")
        
        baseline_results = {}
        for dataset_name in ['uci', 'german', 'australian']:
            baseline_results[dataset_name] = distillator.train_baseline_decision_tree(dataset_name)
        
        print(f"   ✅ Baseline decision tree training completed")
        for dataset_name, result in baseline_results.items():
            print(f"     • {dataset_name.upper()}: Accuracy: {result['accuracy']:.4f}")
        
        # ========================
        # 5. 全特征知识蒸馏实验
        # ========================
        print(f"\n🌟 Phase 5: All-Feature Knowledge Distillation")
        print(f"   Running all-feature distillation with grid search...")
        
        all_feature_distillation_results = distillator.run_all_feature_distillation(
            dataset_names=['uci', 'german', 'australian'],
            temperature_range=[1, 2, 3, 4, 5],   # Temperature: 1-5 (间隔1)
            alpha_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Alpha: 0.0-1.0 (间隔0.1)
            max_depth_range=[4, 5, 6, 7, 8]  # Depth: 4-8
        )
        
        print(f"   ✅ All-feature knowledge distillation completed")
        
        # ========================
        # 6. Top-k知识蒸馏实验
        # ========================
        print(f"\n🧪 Phase 6: Top-k Knowledge Distillation Experiments")
        print(f"   Running comprehensive distillation with parameter optimization...")
        
        # 获取每个数据集的动态k范围 (从5到每个数据集的特征总数)
        k_ranges = {}
        for dataset_name in ['uci', 'german', 'australian']:
            data_dict = processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            k_ranges[dataset_name] = (5, n_features)
            print(f"   {dataset_name.upper()}: k范围 5 到 {n_features} ({n_features-4} 个值)")
        
        # Top-k特征蒸馏实验
        top_k_distillation_results = distillator.run_comprehensive_distillation(
            dataset_names=['uci', 'german', 'australian'],
            k_ranges=k_ranges,         # 每个数据集的动态k范围
            temperature_range=[1, 2, 3, 4, 5],   # Temperature: 1-5 (间隔1)
            alpha_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Alpha: 0.0-1.0 (间隔0.1)
            max_depth_range=[4, 5, 6, 7, 8]        # Depth: 4-8
        )
        
        print(f"   ✅ Top-k knowledge distillation experiments completed")
        
        # ========================
        # 7. 结果汇总和导出
        # ========================
        print(f"\n📊 Phase 7: Results Analysis and Export")
        print(f"   Generating simplified results...")
        
        result_manager = ResultManager()
        
        # 1. 生成四个模型的性能对比表格
        comparison_excel_path = result_manager.generate_model_comparison_table(
            teacher_models, baseline_results, all_feature_distillation_results, top_k_distillation_results
        )
        
        # 2. SHAP特征重要性可视化已在Phase 3中完成，无需重复生成
        
        # 3. 提取最优全特征蒸馏规则
        all_feature_rules_path = result_manager.extract_best_all_feature_rules(all_feature_distillation_results, processed_data)
        
        # 4. 提取最优Top-k蒸馏规则
        topk_rules_path = result_manager.extract_best_topk_rules(top_k_distillation_results, processed_data)
        
        # 5. 清理不需要的文件 - 已禁用，保留所有文件
        # result_manager.clean_output_files()
        
        print(f"\n🎉 System Execution Completed Successfully!")
        print(f"   📁 生成的核心文件:")
        print(f"   📊 1. 模型性能对比Excel: {comparison_excel_path}")
        print(f"   🎯 2. SHAP特征重要性图: {shap_viz_path}")
        print(f"   🌳 3. 全特征蒸馏规则txt: {all_feature_rules_path}")
        print(f"   🌲 4. Top-k蒸馏规则txt: {topk_rules_path}")
        print(f"   📈 5. 全特征消融实验结果(Excel+图)已生成")
        print(f"   📊 6. Top-k消融实验结果(Excel+图)已生成")
        
        # 显示最优蒸馏配置信息
        print(f"\n🏆 最优配置总结:")
        
        # 显示全特征蒸馏的最优配置
        print(f"   🌟 全特征蒸馏最优配置:")
        for dataset_name in ['uci', 'german', 'australian']:
            if dataset_name in all_feature_distillation_results:
                best_config = result_manager._find_best_all_feature_config(all_feature_distillation_results[dataset_name])
                if best_config:
                    print(f"     • {dataset_name.upper()}数据集:")
                    print(f"       - 参数: T={best_config.get('temperature', 'N/A')}, "
                          f"α={best_config.get('alpha', 'N/A')}, D={best_config.get('max_depth', 'N/A')}")
                    print(f"       - 性能: Accuracy={best_config.get('accuracy', 0):.4f}, F1={best_config.get('f1', 0):.4f}")
        
        # 显示Top-k蒸馏的最优配置
        print(f"   🧪 Top-k蒸馏最优配置:")
        for dataset_name in ['uci', 'german', 'australian']:
            if dataset_name in top_k_distillation_results:
                best_config = result_manager._find_best_topk_config(top_k_distillation_results[dataset_name])
                if best_config:
                    print(f"     • {dataset_name.upper()}数据集:")
                    print(f"       - 参数: k={best_config.get('k', 'N/A')}, T={best_config.get('temperature', 'N/A')}, "
                          f"α={best_config.get('alpha', 'N/A')}, D={best_config.get('max_depth', 'N/A')}")
                    print(f"       - 性能: Accuracy={best_config.get('accuracy', 0):.4f}, F1={best_config.get('f1', 0):.4f}")
        
    except Exception as e:
        print(f"\n❌ Error during system execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    main()
