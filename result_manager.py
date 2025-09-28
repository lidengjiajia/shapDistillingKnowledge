"""
精简结果管理器
Simplified Result Manager

只生成三个核心输出：
1. 四个模型在各个数据集上的各个指标表格
2. SHAP值排序图
3. 最优topk规则提取结果
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pickle


class ResultManager:
    """精简结果管理器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    
    def generate_model_comparison_table(self, teacher_models, baseline_results, 
                                      all_feature_results, topk_results):
        """
        生成四个模型的性能对比表格
        
        Args:
            teacher_models: 教师模型结果
            baseline_results: 基线决策树结果  
            all_feature_results: 全特征蒸馏结果
            topk_results: Top-k特征蒸馏结果
        
        Returns:
            str: 保存的Excel文件路径
        """
        print("📊 生成模型性能对比表格...")
        
        comparison_data = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            # 1. 教师模型 (PyTorch神经网络)
            if dataset_name in teacher_models:
                teacher_result = teacher_models[dataset_name]
                comparison_data.append({
                    'Dataset': dataset_name.upper(),
                    'Model_Type': 'Teacher_Model',
                    'Architecture': 'PyTorch_DNN',
                    'Accuracy': teacher_result.get('accuracy', 0),
                    'Precision': teacher_result.get('precision', 0),
                    'Recall': teacher_result.get('recall', 0),
                    'F1_Score': teacher_result.get('f1', 0),
                    'AUC': teacher_result.get('auc', 0)
                })
            
            # 2. 基线决策树
            if dataset_name in baseline_results:
                baseline_result = baseline_results[dataset_name]
                comparison_data.append({
                    'Dataset': dataset_name.upper(),
                    'Model_Type': 'Baseline_DecisionTree',
                    'Architecture': 'Decision_Tree',
                    'Accuracy': baseline_result.get('accuracy', 0),
                    'Precision': baseline_result.get('precision', 0),
                    'Recall': baseline_result.get('recall', 0),
                    'F1_Score': baseline_result.get('f1', 0),
                    'AUC': baseline_result.get('auc', 0)
                })
            
            # 3. 全特征知识蒸馏
            if dataset_name in all_feature_results:
                dataset_results = all_feature_results[dataset_name]
                
                # 处理不同的结果结构
                if isinstance(dataset_results, dict) and 'best' in dataset_results:
                    # 简化结构：{dataset: {'best': result}}
                    best_all_feature = dataset_results['best']
                else:
                    # 复杂结构：需要遍历
                    best_all_feature = self._extract_best_result(dataset_results)
                
                if best_all_feature:
                    comparison_data.append({
                        'Dataset': dataset_name.upper(),
                        'Model_Type': 'All_Feature_Distillation',
                        'Architecture': 'Distilled_DecisionTree',
                        'Accuracy': best_all_feature.get('accuracy', 0),
                        'Precision': best_all_feature.get('precision', 0),
                        'Recall': best_all_feature.get('recall', 0),
                        'F1_Score': best_all_feature.get('f1', 0),
                        'AUC': best_all_feature.get('auc', 0)
                    })
            
            # 4. Top-k特征知识蒸馏
            if dataset_name in topk_results:
                dataset_results = topk_results[dataset_name]
                
                # 处理不同的结果结构
                if isinstance(dataset_results, dict) and 'best' in dataset_results:
                    # 简化结构：{dataset: {'best': result}}
                    best_topk = dataset_results['best']
                else:
                    # 复杂结构：需要遍历
                    best_topk = self._extract_best_topk_result(dataset_results)
                
                if best_topk:
                    comparison_data.append({
                        'Dataset': dataset_name.upper(),
                        'Model_Type': 'TopK_Feature_Distillation',
                        'Architecture': 'Distilled_DecisionTree',
                        'Accuracy': best_topk.get('accuracy', 0),
                        'Precision': best_topk.get('precision', 0),
                        'Recall': best_topk.get('recall', 0),
                        'F1_Score': best_topk.get('f1', 0),
                        'AUC': best_topk.get('auc', 0)
                    })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(comparison_data)
        
        # 格式化数值
        numeric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
        for col in numeric_cols:
            df[col] = df[col].round(4)
        
        # 保存到Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join(self.results_dir, f'model_comparison_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Model_Comparison', index=False)
        
        print(f"   ✅ 模型对比表格已保存：{excel_path}")
        return excel_path
    
    def generate_shap_visualization(self, shap_results):
        """
        DEPRECATED: SHAP可视化现在由shap_analyzer.create_combined_shap_visualization()处理
        该方法返回空路径以保持向后兼容性
        """
        print("📈 SHAP visualization already generated by SHAP analyzer...")
        return None
    
    def extract_best_all_feature_rules(self, all_feature_results, processed_data):
        """
        提取最优全特征蒸馏规则
        
        Args:
            all_feature_results: 全特征蒸馏结果
            processed_data: 预处理后的数据
            
        Returns:
            str: 保存的规则文件路径
        """
        print("🌳 提取最优全特征决策树规则...")
        
        rules_data = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            if dataset_name in all_feature_results:
                best_config = all_feature_results[dataset_name].get('best')
                
                if best_config:
                    # 提取决策规则文本 (注意：rules是字典格式)
                    rules_obj = best_config.get('rules', {})
                    if isinstance(rules_obj, dict) and 'rules' in rules_obj:
                        # rules_obj是字典，包含rules列表
                        tree_rules = '\n'.join(rules_obj['rules']) if rules_obj['rules'] else "规则提取失败"
                    elif isinstance(rules_obj, list):
                        # rules_obj直接是列表
                        tree_rules = '\n'.join(rules_obj) if rules_obj else "规则提取失败"
                    elif isinstance(rules_obj, str):
                        # rules_obj是字符串
                        tree_rules = rules_obj
                    else:
                        tree_rules = "规则格式错误"
                    
                    rules_data.append({
                        'dataset': dataset_name.upper(),
                        'accuracy': best_config.get('accuracy', 0),
                        'f1_score': best_config.get('f1', 0),
                        'precision': best_config.get('precision', 0),
                        'recall': best_config.get('recall', 0),
                        'alpha': best_config.get('alpha', 'N/A'),
                        'temperature': best_config.get('temperature', 'N/A'),
                        'max_depth': best_config.get('max_depth', 'N/A'),
                        'tree_rules': tree_rules
                    })
        
        if not rules_data:
            print("❌ 没有找到有效的全特征蒸馏规则")
            return None
        
        # 保存规则到文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        rules_file = os.path.join(self.results_dir, f'best_all_feature_rules_{timestamp}.txt')
        
        with open(rules_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("最优全特征知识蒸馏决策树规则\n")
            f.write("Best All-Feature Knowledge Distillation Decision Tree Rules\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for rule_data in rules_data:
                f.write(f"📊 {rule_data['dataset']} Dataset:\n")
                f.write(f"   性能指标:\n")
                f.write(f"     • Accuracy: {rule_data['accuracy']:.4f}\n")
                f.write(f"     • F1-Score: {rule_data['f1_score']:.4f}\n")
                f.write(f"     • Precision: {rule_data['precision']:.4f}\n")
                f.write(f"     • Recall: {rule_data['recall']:.4f}\n")
                f.write(f"   最优参数:\n")
                f.write(f"     • Alpha (α): {rule_data['alpha']}\n")
                f.write(f"     • Temperature (T): {rule_data['temperature']}\n")
                f.write(f"     • Max Depth: {rule_data['max_depth']}\n")
                f.write(f"   决策规则:\n")
                f.write(f"{rule_data['tree_rules']}\n")
                f.write("-"*50 + "\n\n")
        
        print(f"   ✅ 全特征蒸馏规则已保存：{rules_file}")
        return rules_file
    
    def extract_best_topk_rules(self, topk_results, processed_data):
        """
        提取最优Top-k规则
        
        Args:
            topk_results: Top-k蒸馏结果
            processed_data: 预处理后的数据
            
        Returns:
            str: 保存的规则文件路径
        """
        print("🌳 提取最优Top-k决策树规则...")
        
        rules_data = []
        
        for dataset_name in ['uci', 'german', 'australian']:
            if dataset_name in topk_results:
                # 找到最佳配置
                best_config = self._find_best_topk_config(topk_results[dataset_name])
                
                if best_config:
                    # 提取决策规则文本
                    tree_rules = best_config.get('tree_rules', 'N/A')
                    if tree_rules == 'N/A' or not tree_rules:
                        # 尝试从rules字段重新构建
                        rules_obj = best_config.get('rules', {})
                        if isinstance(rules_obj, dict) and 'rules' in rules_obj:
                            tree_rules = '\n'.join(rules_obj['rules']) if isinstance(rules_obj['rules'], list) else str(rules_obj['rules'])
                        else:
                            tree_rules = '无法提取决策规则'
                    
                    rules_data.append({
                        'Dataset': dataset_name.upper(),
                        'Best_K': best_config.get('k', 'N/A'),
                        'Best_Temperature': best_config.get('temperature', 'N/A'),
                        'Best_Alpha': best_config.get('alpha', 'N/A'),
                        'Best_Depth': best_config.get('max_depth', 'N/A'),
                        'Accuracy': best_config.get('accuracy', 0),
                        'F1_Score': best_config.get('f1', 0),
                        'Selected_Features': ', '.join(best_config.get('selected_features', [])),
                        'Tree_Rules': tree_rules
                    })
        
        # 保存到文本文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rules_path = os.path.join(self.results_dir, f'best_topk_rules_{timestamp}.txt')
        
        with open(rules_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("最优Top-k特征知识蒸馏决策树规则\n")
            f.write("Best Top-k Feature Knowledge Distillation Decision Tree Rules\n")
            f.write("=" * 80 + "\n\n")
            
            for rule_data in rules_data:
                f.write(f"数据集: {rule_data['Dataset']}\n")
                f.write(f"最佳配置: k={rule_data['Best_K']}, T={rule_data['Best_Temperature']}, "
                       f"α={rule_data['Best_Alpha']}, depth={rule_data['Best_Depth']}\n")
                f.write(f"性能: Accuracy={rule_data['Accuracy']:.4f}, F1={rule_data['F1_Score']:.4f}\n")
                f.write(f"选择特征: {rule_data['Selected_Features']}\n")
                f.write(f"决策规则:\n{rule_data['Tree_Rules']}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"   ✅ 最优Top-k规则已保存：{rules_path}")
        return rules_path
    
    def clean_output_files(self):
        """清理旧的输出文件，只保留最新的重要核心文件"""
        print("🧹 清理旧的输出文件...")
        
        # 获取当前时间戳模式，用于识别当前实验的文件
        current_date = datetime.now().strftime('%Y%m%d')
        
        # 核心保留文件模式（当前实验生成的重要文件）
        core_keep_patterns = [
            'shap_feature_importance.png',         # SHAP图 (无时间戳)
        ]
        
        # 需要保留最新的文件模式（基于时间戳）
        timestamped_keep_patterns = [
            'model_comparison_',                    # 模型对比表格
            'best_all_feature_rules_',             # 全特征规则文件  
            'best_topk_rules_',                    # Top-k规则文件
            'ablation_study_analysis_',            # 全特征消融实验图
            'ablation_study_results_',             # 全特征消融实验Excel
            'topk_ablation_study_analysis_',       # Top-k消融实验图
            'topk_ablation_study_results_',        # Top-k消融实验Excel
        ]
        
        # 旧文件清理模式（这些文件可以安全删除）
        old_file_patterns = [
            'simplified_results_',                 # 旧的简化结果
            'master_results_table_',               # 旧的主结果表
            'teacher_model_',                      # 教师模型文件（pkl/pth）
            'processed_data.pkl',                  # 预处理数据缓存
            'distillation_results.pkl',           # 蒸馏结果缓存  
            'shap_results.pkl',                    # SHAP结果缓存
            'tree_text_',                          # 决策树文本文件
            'comprehensive_model_comparison.xlsx', # 旧的综合对比文件
            'decision_tree_rules_analysis.xlsx'    # 旧的决策树分析文件
        ]
        
        deleted_count = 0
        for filename in os.listdir(self.results_dir):
            file_path = os.path.join(self.results_dir, filename)
            if not os.path.isfile(file_path):
                continue
                
            # 检查是否是核心保留文件
            is_core_file = any(pattern in filename for pattern in core_keep_patterns)
            if is_core_file:
                continue
                
            # 检查是否是当前日期的时间戳文件（保留今天生成的）
            is_current_timestamped = False
            for pattern in timestamped_keep_patterns:
                if pattern in filename and current_date in filename:
                    is_current_timestamped = True
                    break
            if is_current_timestamped:
                continue
                
            # 检查是否是可以删除的旧文件
            should_delete = any(pattern in filename for pattern in old_file_patterns)
            
            # 或者是过期的时间戳文件（不是今天的）
            is_old_timestamped = False
            for pattern in timestamped_keep_patterns:
                if pattern in filename and current_date not in filename:
                    is_old_timestamped = True
                    break
                    
            if should_delete or is_old_timestamped:
                try:
                    os.remove(file_path)
                    print(f"   删除旧文件：{filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"   删除文件失败 {filename}: {e}")
        
        if deleted_count == 0:
            print("   没有发现需要清理的旧文件")
    
    def _extract_best_result(self, results):
        """从结果中提取最佳模型"""
        if not results:
            return None
        
        # 如果results直接是一个字典且包含评估指标，直接返回
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 如果results是模型对象，返回None（无法直接提取指标）
        if not isinstance(results, dict):
            return None
        
        best_accuracy = -1
        best_result = None
        
        # 处理嵌套字典结构
        try:
            # 遍历所有配置找最佳Accuracy（与消融实验报告保持一致）
            for temp, alpha_results in results.items():
                if not isinstance(alpha_results, dict):
                    continue
                for alpha, depth_results in alpha_results.items():
                    if not isinstance(depth_results, dict):
                        continue
                    for depth, result in depth_results.items():
                        if isinstance(result, dict) and 'accuracy' in result:
                            if result['accuracy'] > best_accuracy:
                                best_accuracy = result['accuracy']
                                best_result = result
        except AttributeError:
            # 如果遍历失败，返回None
            return None
        
        return best_result
    
    def _extract_best_topk_result(self, results):
        """从Top-k结果中提取最佳模型"""
        if not results:
            return None
        
        # 如果results直接是一个字典且包含评估指标，直接返回
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 如果results不是字典，返回None
        if not isinstance(results, dict):
            return None
        
        best_accuracy = -1
        best_result = None
        
        try:
            # 遍历所有k值和配置找最佳Accuracy（与消融实验报告保持一致）
            for k, temp_results in results.items():
                if not isinstance(temp_results, dict):
                    continue
                for temp, alpha_results in temp_results.items():
                    if not isinstance(alpha_results, dict):
                        continue
                    for alpha, depth_results in alpha_results.items():
                        if not isinstance(depth_results, dict):
                            continue
                        for depth, result in depth_results.items():
                            if isinstance(result, dict) and 'accuracy' in result:
                                if result['accuracy'] > best_accuracy:
                                    best_accuracy = result['accuracy']
                                    best_result = result
        except (AttributeError, TypeError):
            return None
        
        return best_result
        
        return best_result
    
    def _find_best_all_feature_config(self, results):
        """找到最佳全特征蒸馏配置的详细信息"""
        if not results:
            return None
        
        # 全特征蒸馏结果结构相对简单，直接返回best配置
        if isinstance(results, dict) and 'best' in results:
            return results['best']
        
        return None
    
    def _find_best_topk_config(self, results):
        """找到最佳Top-k配置的详细信息"""
        if not results:
            return None
        
        # 如果results直接是最佳结果字典（包含best键）
        if isinstance(results, dict) and 'best' in results:
            best_result = results['best'].copy()
            # 添加k值
            if 'best_k' in results:
                best_result['k'] = results['best_k']
            
            # 提取决策规则 - 修复规则提取逻辑
            if 'rules' in best_result:
                rules_obj = best_result['rules']
                if isinstance(rules_obj, dict):
                    if 'rules' in rules_obj and isinstance(rules_obj['rules'], list):
                        # 如果rules是列表格式，直接连接
                        best_result['tree_rules'] = '\n'.join(rules_obj['rules'])
                    elif 'description' in rules_obj:
                        # 如果只有描述，使用描述
                        best_result['tree_rules'] = rules_obj['description']
                    else:
                        # 其他情况转换为字符串
                        best_result['tree_rules'] = str(rules_obj)
                elif isinstance(rules_obj, list):
                    # 如果rules直接是列表
                    best_result['tree_rules'] = '\n'.join(rules_obj)
                else:
                    # 其他类型转换为字符串
                    best_result['tree_rules'] = str(rules_obj)
            else:
                best_result['tree_rules'] = 'No rules extracted'
            
            return best_result
        
        # 如果results直接是一个包含指标的字典
        if isinstance(results, dict) and ('f1' in results or 'accuracy' in results):
            return results
        
        # 处理嵌套结构
        if not isinstance(results, dict):
            return None
        
        best_f1 = -1
        best_config = None
        
        try:
            for k, temp_results in results.items():
                if not isinstance(temp_results, dict):
                    continue
                for temp, alpha_results in temp_results.items():
                    if not isinstance(alpha_results, dict):
                        continue
                    for alpha, depth_results in alpha_results.items():
                        if not isinstance(depth_results, dict):
                            continue
                        for depth, result in depth_results.items():
                            if isinstance(result, dict) and 'f1' in result:
                                if result['f1'] > best_f1:
                                    best_f1 = result['f1']
                                    best_config = result.copy()
                                    best_config.update({
                                        'k': k,
                                        'temperature': temp,
                                        'alpha': alpha,
                                        'max_depth': depth
                                    })
                                    # 提取决策规则
                                    if 'rules' in result and isinstance(result['rules'], dict):
                                        if 'rules' in result['rules']:
                                            best_config['tree_rules'] = '\n'.join(result['rules']['rules'])
                                        else:
                                            best_config['tree_rules'] = str(result['rules'])
        except (AttributeError, TypeError):
            return None
        
        return best_config