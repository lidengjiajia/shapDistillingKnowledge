"""
信用评分模型优化系统 - 精简版主程序
只保存核心结果：基线模型、蒸馏模型、SHAP图、消融实验时间
"""

import os
import warnings
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import time
import random

# 设置随机种子
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_global_seed(42)

# 导入自定义模块
from data_preprocessing import DataPreprocessor
from neural_models import train_all_teacher_models
from baseline_models import train_all_baseline_models, save_baseline_results_to_excel
from shap_analysis import SHAPAnalyzer
from distillation_module import KnowledgeDistillator

warnings.filterwarnings('ignore')

def main():
    """精简版主函数 - 只保存核心结果"""
    
    print("="*80)
    print("🎯 信用评分模型优化系统 - 精简版")
    print("="*80)
    
    # 创建结果目录
    os.makedirs('results', exist_ok=True)
    
    # 记录时间
    time_log = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # ========================
        # 1. 数据预处理
        # ========================
        print(f"\n🔄 Phase 1: Data Preprocessing")
        start_time = time.time()
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.process_all_datasets()
        
        time_log['data_preprocessing'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['data_preprocessing']:.2f}s)")
        
        # ========================
        # 2. 基线模型训练（保存到Excel）
        # ========================
        print(f"\n🔧 Phase 2: Baseline Models Training")
        start_time = time.time()
        
        baseline_results_all, baseline_trainer = train_all_baseline_models(processed_data)
        # Excel已在train_all_baseline_models中自动保存
        
        time_log['baseline_training'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['baseline_training']:.2f}s)")
        
        # ========================
        # 3. 神经网络教师模型训练
        # ========================
        print(f"\n🧠 Phase 3: Teacher Models Training")
        start_time = time.time()
        
        teacher_models = train_all_teacher_models(processed_data)
        
        time_log['teacher_training'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['teacher_training']:.2f}s)")
        
        # ========================
        # 4. SHAP分析（保存4个数据集的SHAP图）
        # ========================
        print(f"\n🔍 Phase 4: SHAP Feature Importance Analysis")
        start_time = time.time()
        
        shap_analyzer = SHAPAnalyzer(processed_data)
        shap_analyzer.train_decision_trees()
        
        # 为每个数据集生成SHAP图
        shap_files = []
        for dataset_name in ['german', 'australian', 'uci', 'xinwang']:
            data_dict = processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            
            # 计算SHAP值
            shap_results = shap_analyzer.compute_shap_values(
                dataset_name, 
                top_k_range=(5, n_features)
            )
            
            # 生成SHAP可视化
            shap_path = f"results/shap_{dataset_name}_features.png"
            shap_analyzer.visualize_shap_importance(dataset_name, shap_results, save_path=shap_path)
            shap_files.append(shap_path)
            print(f"   📊 {dataset_name.upper()}: {shap_path}")
        
        # 存储所有SHAP结果用于后续蒸馏
        all_shap_results = {}
        for dataset_name in ['german', 'australian', 'uci', 'xinwang']:
            data_dict = processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            all_shap_results[dataset_name] = shap_analyzer.compute_shap_values(
                dataset_name, 
                top_k_range=(5, n_features)
            )
        
        time_log['shap_analysis'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['shap_analysis']:.2f}s)")
        
        # ========================
        # 5. 知识蒸馏实验（保存最优模型决策规则到Excel）
        # ========================
        # 5. 知识蒸馏实验（消融实验）
        # ========================
        print(f"\n🌟 Phase 5: Knowledge Distillation Ablation Study")
        start_time = time.time()
        
        distillator = KnowledgeDistillator(teacher_models, processed_data, all_shap_results)
        
        # Top-k蒸馏实验
        k_ranges = {}
        for dataset_name in ['german', 'australian', 'uci', 'xinwang']:
            data_dict = processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            k_ranges[dataset_name] = (5, n_features)
        
        distillation_results = distillator.run_comprehensive_distillation(
            dataset_names=['german', 'australian', 'uci', 'xinwang'],
            k_ranges=k_ranges,
            temperature_range=[1, 2, 3, 4, 5],
            alpha_range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            max_depth_range=[4, 5, 6, 7, 8]
        )
        
        time_log['distillation'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['distillation']:.2f}s)")
        
        # ========================
        # 6. 4种模型对比实验
        # ========================
        print(f"\n🔬 Phase 6: Four-Model Comparison")
        start_time = time.time()
        
        # 提取每个数据集的最佳参数
        best_params = {}
        for dataset_name, results in distillation_results.items():
            if 'best' in results:
                best_result = results['best']
                if best_result is None:
                    print(f"[警告] 数据集 {dataset_name} 未找到最佳参数，所有实验均失败或无效。")
                best_params[dataset_name] = {
                    'k': results.get('best_k', 10),
                    'temperature': (best_result or {}).get('temperature', 3.0),
                    'alpha': (best_result or {}).get('alpha', 0.5),
                    'max_depth': (best_result or {}).get('max_depth', 5)
                }
        
        # 运行4种模型对比
        comparison_results = distillator.run_four_model_comparison(
            dataset_names=['german', 'australian', 'uci', 'xinwang'],
            best_params=best_params
        )
        
        # 保存4种模型对比结果到Excel
        four_model_excel = distillator.save_four_model_comparison_to_excel(comparison_results, timestamp)
        
        time_log['four_model_comparison'] = time.time() - start_time
        print(f"   ✅ 完成 ({time_log['four_model_comparison']:.2f}s)")
        
        # ========================
        # 7. 保存时间统计到Excel
        # ========================
        save_time_log_to_excel(time_log, timestamp)
        
        # ========================
        # 总结
        # ========================
        print(f"\n{'='*80}")
        print(f"🎉 所有任务完成！")
        print(f"{'='*80}")
        print(f"📁 生成的文件:")
        print(f"   1️⃣  基线模型结果: results/baseline_models_comparison_{timestamp}.xlsx")
        print(f"   2️⃣  4种模型对比结果: {four_model_excel}")
        print(f"   3️⃣  SHAP特征图 (4个): results/shap_*_features.png")
        print(f"   4️⃣  消融实验数据: results/topk_ablation_study_{timestamp}.csv")
        print(f"   5️⃣  运行时间统计: results/time_log_{timestamp}.xlsx")
        print(f"{'='*80}")
        print(f"\n📊 4种模型对比说明:")
        print(f"   • Baseline Decision Tree - 原始决策树（无蒸馏）")
        print(f"   • Teacher Neural Network - 神经网络教师模型")
        print(f"   • FKD - 全特征知识蒸馏")
        print(f"   • SHAP-KD - Top-k特征知识蒸馏")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


def save_time_log_to_excel(time_log, timestamp):
    """保存运行时间统计到Excel"""
    filename = f"results/time_log_{timestamp}.xlsx"
    
    data = []
    for phase, duration in time_log.items():
        data.append({
            'Phase': phase,
            'Duration_seconds': duration,
            'Duration_minutes': duration / 60
        })
    
    df = pd.DataFrame(data)
    
    # 添加总时间
    total_time = sum(time_log.values())
    df = pd.concat([df, pd.DataFrame([{
        'Phase': 'TOTAL',
        'Duration_seconds': total_time,
        'Duration_minutes': total_time / 60
    }])], ignore_index=True)
    
    df.to_excel(filename, index=False)
    print(f"   ⏱️  运行时间已保存: {filename}")
    
    return filename


if __name__ == "__main__":
    main()
