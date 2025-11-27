# Knowledge Distillation Module - Decision Tree Only
# 知识蒸馏模块 - 仅决策树版本

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
from datetime import datetime
import pandas as pd

# 并发配置：使用CPU核心数-1，至少为1
n_jobs = max(1, min(os.cpu_count() - 1, os.cpu_count()))
# 只在需要时显示配置信息，避免重复输出

# 导入消融实验分析器
from ablation_analyzer import ablation_analyzer

# 创建Top-k消融分析器的全局实例
topk_ablation_analyzer = None

warnings.filterwarnings('ignore')

# 设置matplotlib后端为非交互式，避免多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 禁用Optuna日志输出

class KnowledgeDistillator:
    """知识蒸馏系统 - 决策树蒸馏"""
    
    def __init__(self, teacher_models, processed_data, all_shap_results):
        self.teacher_models = teacher_models
        self.processed_data = processed_data
        self.all_shap_results = all_shap_results
        
    def extract_knowledge(self, dataset_name, model_type, temperature=3.0):
        """从教师模型提取知识
        
        知识蒸馏理论背景：
        教师模型输出softmax分布包含更丰富的类间关系信息
        temperature参数控制分布的平滑程度，温度越高分布越平滑
        """
        teacher_model = self.teacher_models[dataset_name]['model']
        data_dict = self.processed_data[dataset_name]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        
        # 获取教师模型的软标签(概率分布)
        train_logits = self._get_teacher_predictions(teacher_model, X_train)
        test_logits = self._get_teacher_predictions(teacher_model, X_test)
        
        # 应用温度缩放，增强知识蒸馏效果
        train_soft_labels = self._apply_temperature(train_logits, temperature)
        test_soft_labels = self._apply_temperature(test_logits, temperature)
        
        return {
            'train_soft_labels': train_soft_labels,
            'test_soft_labels': test_soft_labels,
            'teacher_logits_train': train_logits,
            'teacher_logits_test': test_logits
        }
    
    def _get_teacher_predictions(self, teacher_model, X):
        """从教师模型获取预测概率 - 兼容PyTorch和sklearn模型"""
        import torch
        
        # 检查是否是PyTorch模型
        if hasattr(teacher_model, 'eval') and hasattr(teacher_model, 'forward'):
            # PyTorch模型
            teacher_model.eval()
            device = next(teacher_model.parameters()).device
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = teacher_model(X_tensor)
                # 对于二分类，将sigmoid输出转换为两类概率
                probs_class1 = outputs.cpu().numpy().flatten()
                probs_class0 = 1 - probs_class1
                return np.column_stack([probs_class0, probs_class1])
        else:
            # sklearn模型
            return teacher_model.predict_proba(X)
        
        return {
            'train_soft_labels': train_soft_labels,
            'test_soft_labels': test_soft_labels,
            'teacher_logits_train': train_logits,
            'teacher_logits_test': test_logits
        }
    
    def _apply_temperature(self, logits, temperature):
        """温度缩放：logits / T，然后应用softmax
        温度T > 1 使分布更平滑，T < 1 使分布更sharp
        """
        return F.softmax(torch.tensor(logits) / temperature, dim=1).numpy()
    
    def train_student_model(self, dataset_name, model_type_name='decision_tree', 
                          k=5, temperature=3.0, alpha=0.7, max_depth=6, 
                          use_all_features=False, trial=None):
        """训练学生模型(决策树)使用知识蒸馏
        
        参数:
        - dataset_name: 数据集名称
        - model_type_name: 学生模型类型，固定为'decision_tree'
        - k: Top-k特征数量
        - temperature: 知识蒸馏温度参数
        - alpha: 蒸馏损失权重 (0=仅硬标签, 1=仅软标签)
        - max_depth: 决策树最大深度
        - use_all_features: 是否使用全特征
        - trial: Optuna trial对象(用于超参数优化)
        """
        
        data_dict = self.processed_data[dataset_name]
        
        # 特征选择
        if use_all_features:
            # 使用全特征
            X_train_selected = data_dict['X_train']
            X_test_selected = data_dict['X_test']
            selected_features = data_dict['feature_names']
            model_type = f'all_features_decision_tree_distillation'
        else:
            # 选择Top-k特征
            shap_results = self.all_shap_results[dataset_name]
            top_k_features = shap_results['top_k_features'][k]
            feature_indices = [data_dict['feature_names'].index(feat) for feat in top_k_features]
            
            X_train_selected = data_dict['X_train'][:, feature_indices]
            X_test_selected = data_dict['X_test'][:, feature_indices]
            selected_features = top_k_features
            model_type = f'top_{k}_decision_tree_distillation'
        
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # 提取教师模型知识
        knowledge = self.extract_knowledge(dataset_name, 'teacher', temperature)
        train_soft_labels = knowledge['train_soft_labels']
        test_soft_labels = knowledge['test_soft_labels']
        
        # 创建决策树学生模型
        student_model = self._create_decision_tree_student(trial, max_depth)
        
        # 知识蒸馏训练
        student_model = self._train_with_distillation(
            student_model, X_train_selected, y_train, train_soft_labels, alpha
        )
        
        # 预测和评估
        y_pred = student_model.predict(X_test_selected)
        y_pred_proba = student_model.predict_proba(X_test_selected)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 提取决策规则
        rules = self._extract_decision_rules(student_model, selected_features)
        
        return {
            'model': student_model,
            'model_type': model_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_count': len(selected_features),
            'selected_features': selected_features,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'rules': rules,
            'temperature': temperature,
            'alpha': alpha,
            'max_depth': max_depth,
            'hyperparameters': {
                'temperature': temperature,
                'alpha': alpha,
                'max_depth': max_depth
            }
        }
    
    def _create_decision_tree_student(self, trial, max_depth):
        """创建决策树学生模型"""
        if trial is not None:
            # Optuna超参数优化
            trial_max_depth = trial.suggest_int('max_depth', 3, 12)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        else:
            # 使用固定参数
            trial_max_depth = max_depth
            min_samples_split = 2
            min_samples_leaf = 1
            max_features = 'sqrt'
        
        return DecisionTreeClassifier(
            max_depth=trial_max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
    
    def _train_with_distillation(self, model, X_train, y_train, soft_labels, alpha):
        """使用知识蒸馏训练决策树
        
        对于决策树，我们使用软标签的概率作为样本权重
        这是一种近似的知识蒸馏方法，因为决策树不直接支持软标签
        """
        
        if alpha > 0:
            # 使用软标签的最大概率作为样本权重
            sample_weights = np.max(soft_labels, axis=1)
            # 归一化权重
            sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
            
            # 训练时使用样本权重
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            # 纯硬标签训练
            model.fit(X_train, y_train)
        
        return model
    
    def _extract_decision_rules(self, model, feature_names):
        """提取决策树规则"""
        # 简化规则提取，不依赖外部模块
        rules = self._simple_extract_rules(model, feature_names)
        
        return {
            'rules': rules,
            'rule_count': len(rules),
            'description': f'Decision tree with {len(rules)} rules'
        }
    
    def _simple_extract_rules(self, model, feature_names):
        """简单的决策树规则提取"""
        tree = model.tree_
        rules = []
        
        def recurse(node, depth, parent_rule=""):
            if tree.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]
                left_rule = f"{parent_rule}{name} <= {threshold:.4f}"
                right_rule = f"{parent_rule}{name} > {threshold:.4f}"
                recurse(tree.children_left[node], depth + 1, left_rule + " and ")
                recurse(tree.children_right[node], depth + 1, right_rule + " and ")
            else:
                # 叶子节点
                if parent_rule:
                    rule = parent_rule.rstrip(" and ")
                    value = tree.value[node]
                    predicted_class = np.argmax(value)
                    confidence = np.max(value) / np.sum(value)
                    rules.append(f"IF {rule} THEN class={predicted_class} (confidence={confidence:.4f})")
        
        try:
            recurse(0, 0)
        except Exception as e:
            # 如果规则提取失败，返回简单描述
            rules = [f"Decision tree with {tree.node_count} nodes"]
        
        return rules
    
    def train_baseline_decision_tree(self, dataset_name, max_depth=5):
        """训练基础决策树（不使用蒸馏）
        这是4种模型对比中的第1种：原始决策树
        """
        data_dict = self.processed_data[dataset_name]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        
        # 固定参数训练基础决策树（无Optuna）
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_train, y_train)

        # 预测和评估
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 提取决策规则
        rules = self._extract_decision_rules(model, feature_names)

        params = {
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt'
        }

        return {
            'model': model,
            'model_type': 'baseline_tree',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_count': len(feature_names),
            'selected_features': feature_names,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'rules': rules,
            'hyperparameters': params,
            'best_params': params
        }
    
    def run_all_feature_distillation(self, dataset_names, temperature_range, alpha_range, max_depth_range):
        """运行全特征知识蒸馏实验并记录消融实验数据"""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset...")
            results[dataset_name] = {}
            
            best_accuracy = 0  # 改为使用准确率作为评判标准
            best_result = None
            
            total_combinations = len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"🎓 {dataset_name.upper()}", 
                               unit="exp", position=0, leave=True)
            print(f"     全特征实验组合数: {total_combinations}")
            
            for temperature in temperature_range:
                for alpha in alpha_range:
                    for max_depth in max_depth_range:
                        progress_bar.set_postfix({
                            'T': temperature, 
                            'α': f"{alpha:.1f}", 
                            'D': max_depth,
                            'Best': f"{best_accuracy:.4f}"
                        })
                        result = self.train_student_model(
                            dataset_name=dataset_name,
                            model_type_name='decision_tree',
                            use_all_features=True,
                            temperature=temperature,
                            alpha=alpha,
                            max_depth=max_depth
                        )
                        
                        # 记录全特征蒸馏的消融实验数据
                        ablation_analyzer.record_experiment_result(
                            dataset_name=dataset_name,
                            k=None,  # 全特征蒸馏没有k值
                            temperature=temperature,
                            alpha=alpha,
                            max_depth=max_depth,
                            accuracy=result['accuracy'],
                            f1_score=result['f1'],
                            precision=result['precision'],
                            recall=result['recall']
                        )
                        
                        if result['accuracy'] > best_accuracy:  # 改为使用准确率
                            best_accuracy = result['accuracy']
                            best_result = result
                        
                        progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            print(f"     Best Accuracy: {best_accuracy:.4f}")  # 改为显示准确率
        
        # 保存消融实验数据和创建可视化
        print("\n📊 Saving ablation study data and creating visualizations for all-feature distillation...")
        ablation_analyzer.save_ablation_data(prefix='ablation_study')
        ablation_analyzer.create_ablation_visualizations()
        ablation_analyzer.generate_summary_report(prefix='ablation_study')
        
        return results
    
    
    def run_comprehensive_distillation(self, dataset_names, k_ranges, temperature_range, alpha_range, max_depth_range):
        """运行综合知识蒸馏实验（Top-k特征）
        
        Args:
            dataset_names: 数据集名称列表
            k_ranges: 每个数据集的k范围字典 {'german': (5, 54), 'australian': (5, 22), 'uci': (5, 23)}
            temperature_range: 温度参数范围
            alpha_range: 加权参数范围
            max_depth_range: 决策树深度范围
        """
        global topk_ablation_analyzer
        
        # 初始化Top-k消融分析器
        from ablation_analyzer import AblationStudyAnalyzer
        topk_ablation_analyzer = AblationStudyAnalyzer()
        topk_ablation_analyzer.experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {}
        
        for dataset_name in dataset_names:
            print(f"   Processing {dataset_name.upper()} dataset...")
            results[dataset_name] = {}
            
            best_accuracy = 0  # 改为使用准确率作为评判标准
            best_result = None
            best_k = None
            
            # 获取当前数据集的k范围
            dataset_k_range = k_ranges[dataset_name]
            k_values = list(range(dataset_k_range[0], dataset_k_range[1] + 1))
            total_combinations = len(k_values) * len(temperature_range) * len(alpha_range) * len(max_depth_range)
            progress_bar = tqdm(total=total_combinations, desc=f"🔍 {dataset_name.upper()}", 
                               unit="exp", position=0, leave=True)
            print(f"     k范围: {dataset_k_range[0]} 到 {dataset_k_range[1]} ({len(k_values)} 个值)")
            print(f"     Top-k实验组合数: {total_combinations}")
            
            # 准备并发执行的实验参数
            experiment_params = []
            for k in k_values:
                for temperature in temperature_range:
                    for alpha in alpha_range:
                        for max_depth in max_depth_range:
                            experiment_params.append((dataset_name, k, temperature, alpha, max_depth))
            
            # 设置并发数量（使用线程池避免multiprocessing的编码问题）
            import platform
            if platform.system() == 'Windows':
                n_jobs = min(4, max(1, os.cpu_count() // 2))
            else:
                n_jobs = max(1, min(os.cpu_count() - 1, os.cpu_count()))
            
            print(f"     🚀 Using {n_jobs} parallel threads for Top-k distillation")
            
            # 并发执行实验
            def run_single_experiment(params):
                dataset_name, k, temperature, alpha, max_depth = params
                try:
                    result = self.train_student_model(
                        dataset_name=dataset_name,
                        model_type_name='decision_tree',
                        k=k,
                        temperature=temperature,
                        alpha=alpha,
                        max_depth=max_depth,
                        use_all_features=False
                    )
                    return params, result, None
                except Exception as e:
                    return params, None, str(e)
            
            # 使用线程池并行执行（避免multiprocessing的编码问题）
            all_results = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # 提交所有任务
                future_to_params = {
                    executor.submit(run_single_experiment, params): params 
                    for params in experiment_params
                }
                
                # 处理结果
                for future in as_completed(future_to_params):
                    params, result, error = future.result()
                    if error:
                        print(f"     ❌ Error in experiment {params}: {error}")
                        continue
                    
                    dataset_name, k, temperature, alpha, max_depth = params
                    
                    # 记录Top-k蒸馏的消融实验数据
                    topk_ablation_analyzer.record_experiment_result(
                        dataset_name=dataset_name,
                        k=k,
                        temperature=temperature,
                        alpha=alpha,
                        max_depth=max_depth,
                        accuracy=result['accuracy'],
                        f1_score=result['f1'],
                        precision=result['precision'],
                        recall=result['recall']
                    )
                    
                    all_results.append((params, result))
                    
                    if result['accuracy'] > best_accuracy:
                        best_accuracy = result['accuracy']
                        best_result = result
                        best_k = k
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'k': k,
                        'T': temperature, 
                        'α': f"{alpha:.1f}", 
                        'D': max_depth,
                        'Best': f"{best_accuracy:.4f}"
                    })
                    progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            results[dataset_name]['best_k'] = best_k
            print(f"     Best Accuracy: {best_accuracy:.4f} with k={best_k}")  # 改为显示准确率
            
            # 🌲 保存最佳SHAP-KD模型的决策树规则和路径
            if best_result is not None:
                print(f"\n   🌲 Saving best SHAP-KD model rules and paths for {dataset_name.upper()}...")
                data_dict = self.processed_data[dataset_name]
                
                # 保存决策规则
                best_params = {
                    'k': best_k,
                    'temperature': best_result['temperature'],
                    'alpha': best_result['alpha'],
                    'max_depth': best_result['max_depth']
                }
                self._save_decision_tree_rules(
                    model=best_result['model'],
                    dataset_name=dataset_name,
                    model_type='SHAP-KD',
                    feature_names=best_result['selected_features'],
                    params=best_params
                )
                
                # 保存决策路径
                selected_feature_names = best_result['selected_features']
                feature_indices = [data_dict['feature_names'].index(feat) for feat in selected_feature_names]
                X_test_selected = data_dict['X_test'][:, feature_indices]
                
                self._save_decision_tree_paths(
                    model=best_result['model'],
                    dataset_name=dataset_name,
                    model_type='SHAP-KD',
                    X_test=X_test_selected,
                    y_test=data_dict['y_test'],
                    feature_names=selected_feature_names,
                    params=best_params
                )
        
        # 保存Top-k消融实验数据和创建可视化
        print("\n📊 Saving Top-k ablation study data and creating visualizations...")
        topk_ablation_analyzer.save_ablation_data(prefix='topk_ablation_study')
        # 使用通用的消融实验可视化方法（避免重复生成）
        topk_ablation_analyzer.create_ablation_visualizations()
        topk_ablation_analyzer.generate_summary_report(prefix='topk_ablation_study')
        
        return results
    
    def run_four_model_comparison(self, dataset_names, best_params):
        """运行4种模型对比实验
        
        4种模型：
        1. Baseline Decision Tree - 原始决策树（不使用蒸馏）
        2. Teacher Model - 神经网络教师模型
        3. FKD - Full Knowledge Distillation（使用全部特征的知识蒸馏）
        4. SHAP-KD - Top-k Knowledge Distillation（使用SHAP选择的Top-k特征）
        
        Args:
            dataset_names: 数据集名称列表
            best_params: 每个数据集的最佳参数字典
                格式: {'german': {'k': 10, 'temperature': 3, 'alpha': 0.5, 'max_depth': 5}, ...}
        
        Returns:
            comparison_results: 包含4种模型对比结果的字典
        """
        comparison_results = {}
        
        for dataset_name in dataset_names:
            print(f"\n🔬 Running 4-Model Comparison for {dataset_name.upper()}...")
            comparison_results[dataset_name] = {}
            
            data_dict = self.processed_data[dataset_name]
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # 获取该数据集的最佳参数
            params = best_params.get(dataset_name, {
                'k': 10, 'temperature': 3.0, 'alpha': 0.5, 'max_depth': 5
            })
            
            # 1️⃣ 原始决策树 (Baseline Decision Tree)
            print(f"   1️⃣ Training Baseline Decision Tree...")
            baseline_dt_result = self.train_baseline_decision_tree(
                dataset_name, 
                max_depth=params.get('max_depth', 5)
            )
            comparison_results[dataset_name]['baseline_dt'] = {
                'model_name': 'Baseline Decision Tree',
                'accuracy': baseline_dt_result['accuracy'],
                'precision': baseline_dt_result['precision'],
                'recall': baseline_dt_result['recall'],
                'f1': baseline_dt_result['f1'],
                'feature_count': baseline_dt_result['feature_count'],
                'max_depth': params.get('max_depth', 5)
            }
            print(f"      Accuracy: {baseline_dt_result['accuracy']:.4f}")
            
            # 2️⃣ 教师模型 (Teacher Neural Network)
            print(f"   2️⃣ Evaluating Teacher Model...")
            teacher_model = self.teacher_models[dataset_name]['model']
            teacher_pred = self._get_teacher_hard_predictions(teacher_model, X_test)
            teacher_accuracy = accuracy_score(y_test, teacher_pred)
            teacher_precision = precision_score(y_test, teacher_pred, average='weighted', zero_division=0)
            teacher_recall = recall_score(y_test, teacher_pred, average='weighted', zero_division=0)
            teacher_f1 = f1_score(y_test, teacher_pred, average='weighted', zero_division=0)
            
            comparison_results[dataset_name]['teacher'] = {
                'model_name': 'Teacher Neural Network',
                'accuracy': teacher_accuracy,
                'precision': teacher_precision,
                'recall': teacher_recall,
                'f1': teacher_f1,
                'feature_count': len(data_dict['feature_names'])
            }
            print(f"      Accuracy: {teacher_accuracy:.4f}")
            
            # 3️⃣ 全特征知识蒸馏 (Full Knowledge Distillation - FKD)
            print(f"   3️⃣ Training FKD (Full Knowledge Distillation)...")
            fkd_result = self.train_student_model(
                dataset_name=dataset_name,
                model_type_name='decision_tree',
                k=None,  # 不使用k
                temperature=params.get('temperature', 3.0),
                alpha=params.get('alpha', 0.5),
                max_depth=params.get('max_depth', 5),
                use_all_features=True  # 使用全部特征
            )
            comparison_results[dataset_name]['fkd'] = {
                'model_name': 'FKD (All Features)',
                'accuracy': fkd_result['accuracy'],
                'precision': fkd_result['precision'],
                'recall': fkd_result['recall'],
                'f1': fkd_result['f1'],
                'feature_count': fkd_result['feature_count'],
                'temperature': params.get('temperature', 3.0),
                'alpha': params.get('alpha', 0.5),
                'max_depth': params.get('max_depth', 5)
            }
            print(f"      Accuracy: {fkd_result['accuracy']:.4f}")
            
            # 4️⃣ Top-k知识蒸馏 (SHAP-KD)
            print(f"   4️⃣ Training SHAP-KD (Top-{params.get('k', 10)} Features)...")
            shap_kd_result = self.train_student_model(
                dataset_name=dataset_name,
                model_type_name='decision_tree',
                k=params.get('k', 10),
                temperature=params.get('temperature', 3.0),
                alpha=params.get('alpha', 0.5),
                max_depth=params.get('max_depth', 5),
                use_all_features=False  # 使用Top-k特征
            )
            comparison_results[dataset_name]['shap_kd'] = {
                'model_name': f'SHAP-KD (Top-{params.get("k", 10)})',
                'accuracy': shap_kd_result['accuracy'],
                'precision': shap_kd_result['precision'],
                'recall': shap_kd_result['recall'],
                'f1': shap_kd_result['f1'],
                'feature_count': shap_kd_result['feature_count'],
                'k': params.get('k', 10),
                'temperature': params.get('temperature', 3.0),
                'alpha': params.get('alpha', 0.5),
                'max_depth': params.get('max_depth', 5)
            }
            print(f"      Accuracy: {shap_kd_result['accuracy']:.4f}")
            
            # 注意：规则和路径提取已在消融实验的最佳模型中完成，此处不重复保存
            # 如需单独保存四模型对比的规则，可取消下方注释
            # # 🌲 保存SHAP-KD决策树规则
            # print(f"   🌲 Extracting and saving SHAP-KD decision tree rules...")
            # self._save_decision_tree_rules(
            #     model=shap_kd_result['model'],
            #     dataset_name=dataset_name,
            #     model_type='SHAP-KD',
            #     feature_names=shap_kd_result.get('feature_names', data_dict['feature_names'][:params.get('k', 10)]),
            #     params=params
            # )
            # 
            # # 🛤️ 保存SHAP-KD决策树路径（每个样本的具体路径）
            # print(f"   🛤️ Extracting and saving SHAP-KD decision tree paths...")
            # 
            # # 获取SHAP-KD使用的特征索引
            # selected_feature_names = shap_kd_result.get('selected_features', data_dict['feature_names'][:params.get('k', 10)])
            # if isinstance(selected_feature_names, list) and isinstance(selected_feature_names[0], str):
            #     # 如果是特征名称列表，转换为索引
            #     feature_indices = [data_dict['feature_names'].index(feat) for feat in selected_feature_names]
            #     X_test_selected = data_dict['X_test'][:, feature_indices]
            # else:
            #     # 如果已经是索引，直接使用
            #     X_test_selected = data_dict['X_test'][:, :params.get('k', 10)]
            #     selected_feature_names = data_dict['feature_names'][:params.get('k', 10)]
            # 
            # self._save_decision_tree_paths(
            #     model=shap_kd_result['model'],
            #     dataset_name=dataset_name,
            #     model_type='SHAP-KD',
            #     X_test=X_test_selected,
            #     y_test=data_dict['y_test'],
            #     feature_names=selected_feature_names,
            #     params=params
            # )
            
            print(f"\n   ✅ {dataset_name.upper()} Comparison Complete")
            print(f"      Baseline DT: {baseline_dt_result['accuracy']:.4f}")
            print(f"      Teacher: {teacher_accuracy:.4f}")
            print(f"      FKD: {fkd_result['accuracy']:.4f}")
            print(f"      SHAP-KD: {shap_kd_result['accuracy']:.4f}")
        
        return comparison_results
    
    def _get_teacher_hard_predictions(self, teacher_model, X):
        """获取教师模型的硬预测（类别标签）"""
        import torch
        
        # 检查是否是PyTorch模型
        if hasattr(teacher_model, 'eval') and hasattr(teacher_model, 'forward'):
            # PyTorch模型
            teacher_model.eval()
            device = next(teacher_model.parameters()).device
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = teacher_model(X_tensor)
                # 对于二分类，将sigmoid输出转换为类别
                probs = outputs.cpu().numpy().flatten()
                predictions = (probs > 0.5).astype(int)
                return predictions
        else:
            # sklearn模型
            return teacher_model.predict(X)
    
    def save_four_model_comparison_to_excel(self, comparison_results, timestamp):
        """保存4种模型对比结果到Excel
        
        Args:
            comparison_results: run_four_model_comparison返回的结果字典
            timestamp: 时间戳字符串
        
        Returns:
            filename: 保存的Excel文件路径
        """
        filename = f"results/four_model_comparison_{timestamp}.xlsx"
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 为每个数据集创建一个sheet
            for dataset_name, models in comparison_results.items():
                data = []
                for model_key, model_result in models.items():
                    row = {
                        'Model': model_result['model_name'],
                        'Accuracy': f"{model_result['accuracy']:.4f}",
                        'Precision': f"{model_result['precision']:.4f}",
                        'Recall': f"{model_result['recall']:.4f}",
                        'F1_Score': f"{model_result['f1']:.4f}",
                        'Feature_Count': model_result.get('feature_count', 'N/A')
                    }
                    
                    # 添加特定模型的额外参数
                    if 'k' in model_result:
                        row['k'] = model_result['k']
                    if 'temperature' in model_result:
                        row['Temperature'] = model_result['temperature']
                    if 'alpha' in model_result:
                        row['Alpha'] = model_result['alpha']
                    if 'max_depth' in model_result:
                        row['Max_Depth'] = model_result['max_depth']
                    
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=dataset_name.upper(), index=False)
            
            # 创建汇总sheet
            summary_data = []
            for dataset_name, models in comparison_results.items():
                for model_key, model_result in models.items():
                    summary_data.append({
                        'Dataset': dataset_name.upper(),
                        'Model': model_result['model_name'],
                        'Accuracy': f"{model_result['accuracy']:.4f}",
                        'F1_Score': f"{model_result['f1']:.4f}",
                        'Feature_Count': model_result.get('feature_count', 'N/A')
                    })
            
            summary_df = pd.DataFrame(summary_data)
            # 按数据集和准确率排序
            summary_df = summary_df.sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
            summary_df.to_excel(writer, sheet_name='SUMMARY', index=False)
        
        print(f"\n📊 Four-Model Comparison saved to: {filename}")
        return filename
    
    def _save_decision_tree_rules(self, model, dataset_name, model_type, feature_names, params):
        """提取并保存决策树规则到文本文件
        
        Args:
            model: 训练好的决策树模型
            dataset_name: 数据集名称
            model_type: 模型类型 (如 'SHAP-KD', 'FKD', 'Baseline')
            feature_names: 特征名称列表
            params: 模型参数字典
        """
        from sklearn.tree import _tree
        import os
        
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 构建文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{dataset_name}_{model_type}_decision_rules_{timestamp}.txt"
        
        tree = model.tree_
        
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("="*80 + "\n")
            f.write(f"决策树规则提取 - {dataset_name.upper()} 数据集\n")
            f.write(f"模型类型: {model_type}\n")
            f.write("="*80 + "\n\n")
            
            # 写入模型参数
            f.write("模型参数:\n")
            f.write("-"*80 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 写入树的基本信息
            f.write("决策树信息:\n")
            f.write("-"*80 + "\n")
            f.write(f"  节点总数: {tree.node_count}\n")
            f.write(f"  最大深度: {tree.max_depth}\n")
            f.write(f"  叶子节点数: {tree.n_leaves}\n")
            f.write(f"  使用特征数: {len(feature_names)}\n")
            f.write("\n")
            
            # 写入特征名称
            f.write("使用的特征:\n")
            f.write("-"*80 + "\n")
            for i, fname in enumerate(feature_names, 1):
                f.write(f"  {i}. {fname}\n")
            f.write("\n")
            
            # 提取并写入决策规则
            f.write("决策规则:\n")
            f.write("="*80 + "\n\n")
            
            rule_count = [0]  # 使用列表以便在递归函数中修改
            
            def recurse(node, depth, rule_path=""):
                """递归提取规则"""
                indent = "  " * depth
                
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    # 内部节点
                    feature_name = feature_names[tree.feature[node]]
                    threshold = tree.threshold[node]
                    
                    # 左子树 (<=)
                    left_rule = f"{rule_path}{'AND ' if rule_path else ''}({feature_name} <= {threshold:.4f})"
                    recurse(tree.children_left[node], depth + 1, left_rule + " ")
                    
                    # 右子树 (>)
                    right_rule = f"{rule_path}{'AND ' if rule_path else ''}({feature_name} > {threshold:.4f})"
                    recurse(tree.children_right[node], depth + 1, right_rule + " ")
                else:
                    # 叶子节点 - 生成规则
                    rule_count[0] += 1
                    value = tree.value[node][0]
                    predicted_class = np.argmax(value)
                    samples = np.sum(value)
                    confidence = np.max(value) / samples if samples > 0 else 0
                    
                    f.write(f"规则 {rule_count[0]}:\n")
                    f.write(f"{indent}IF {rule_path.strip()}\n")
                    f.write(f"{indent}THEN 预测类别 = {predicted_class}\n")
                    f.write(f"{indent}     置信度 = {confidence:.4f} ({int(np.max(value))}/{int(samples)} 样本)\n")
                    f.write(f"{indent}     样本分布 = {[int(x) for x in value]}\n")
                    f.write("\n")
            
            # 开始递归提取
            try:
                recurse(0, 0)
                f.write("="*80 + "\n")
                f.write(f"总共提取了 {rule_count[0]} 条规则\n")
                f.write("="*80 + "\n")
            except Exception as e:
                f.write(f"\n提取规则时出错: {str(e)}\n")
        
        print(f"      Rules saved to: {filename}")
        return filename
    
    def _save_decision_tree_paths(self, model, dataset_name, model_type, X_test, y_test, feature_names, params):
        """提取并保存每个测试样本通过决策树的路径
        
        Args:
            model: 训练好的决策树模型
            dataset_name: 数据集名称
            model_type: 模型类型 (如 'SHAP-KD')
            X_test: 测试集特征
            y_test: 测试集真实标签
            feature_names: 特征名称列表
            params: 模型参数字典
        """
        from sklearn.tree import _tree
        import os
        
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 构建文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/{dataset_name}_{model_type}_tree_paths_{timestamp}.txt"
        
        tree = model.tree_
        y_pred = model.predict(X_test)
        
        # 获取每个样本的决策路径
        decision_paths = model.decision_path(X_test)
        
        with open(filename, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("="*100 + "\n")
            f.write(f"决策树路径提取 - {dataset_name.upper()} 数据集\n")
            f.write(f"模型类型: {model_type}\n")
            f.write("="*100 + "\n\n")
            
            # 写入模型参数
            f.write("模型参数:\n")
            f.write("-"*100 + "\n")
            for key, value in params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 写入统计信息
            f.write("路径统计:\n")
            f.write("-"*100 + "\n")
            f.write(f"  测试样本总数: {len(X_test)}\n")
            f.write(f"  特征数量: {len(feature_names)}\n")
            f.write(f"  树的最大深度: {tree.max_depth}\n")
            f.write(f"  预测准确率: {accuracy_score(y_test, y_pred):.4f}\n")
            f.write("\n")
            
            # 提取每个样本的路径
            f.write("样本决策路径:\n")
            f.write("="*100 + "\n\n")
            
            # 计算路径长度分布
            path_lengths = []
            
            for sample_idx in range(len(X_test)):
                # 获取该样本经过的节点
                node_indices = decision_paths.indices[decision_paths.indptr[sample_idx]:
                                                     decision_paths.indptr[sample_idx + 1]]
                
                path_length = len(node_indices)
                path_lengths.append(path_length)
                
                # 写入样本信息
                f.write(f"样本 {sample_idx + 1}:\n")
                f.write(f"  真实标签: {int(y_test[sample_idx])}, 预测标签: {int(y_pred[sample_idx])}, ")
                f.write(f"预测结果: {'✓正确' if y_test[sample_idx] == y_pred[sample_idx] else '✗错误'}\n")
                f.write(f"  路径长度: {path_length} (经过 {path_length} 个节点)\n")
                f.write(f"  决策路径:\n")
                
                # 逐个节点追踪路径
                for depth, node_id in enumerate(node_indices):
                    indent = "    " + "  " * depth
                    
                    if tree.feature[node_id] != _tree.TREE_UNDEFINED:
                        # 内部节点
                        feature_name = feature_names[tree.feature[node_id]]
                        threshold = tree.threshold[node_id]
                        feature_value = X_test[sample_idx, tree.feature[node_id]]
                        
                        # 判断走向
                        if feature_value <= threshold:
                            direction = "左分支 (≤)"
                            symbol = "✓"
                        else:
                            direction = "右分支 (>)"
                            symbol = "✗"
                        
                        f.write(f"{indent}节点 {node_id}: [{feature_name}] = {feature_value:.4f} ")
                        f.write(f"{symbol} (阈值: {threshold:.4f}) → {direction}\n")
                    else:
                        # 叶子节点
                        value = tree.value[node_id][0]
                        predicted_class = np.argmax(value)
                        samples = np.sum(value)
                        confidence = np.max(value) / samples if samples > 0 else 0
                        
                        f.write(f"{indent}叶子节点 {node_id}: 预测类别 = {predicted_class}, ")
                        f.write(f"置信度 = {confidence:.4f}, 样本分布 = {[int(x) for x in value]}\n")
                
                f.write("\n")
                
                # 每50个样本输出一次进度
                if (sample_idx + 1) % 50 == 0:
                    f.write(f"--- 已处理 {sample_idx + 1}/{len(X_test)} 个样本 ---\n\n")
            
            # 写入路径统计分析
            f.write("="*100 + "\n")
            f.write("路径长度分析:\n")
            f.write("-"*100 + "\n")
            f.write(f"  平均路径长度: {np.mean(path_lengths):.2f}\n")
            f.write(f"  最短路径: {np.min(path_lengths)}\n")
            f.write(f"  最长路径: {np.max(path_lengths)}\n")
            f.write(f"  路径长度标准差: {np.std(path_lengths):.2f}\n")
            f.write("\n")
            
            # 按路径长度统计样本数
            unique_lengths, counts = np.unique(path_lengths, return_counts=True)
            f.write("路径长度分布:\n")
            for length, count in zip(unique_lengths, counts):
                percentage = count / len(path_lengths) * 100
                f.write(f"  长度 {length}: {count} 个样本 ({percentage:.2f}%)\n")
            
            f.write("="*100 + "\n")
            f.write(f"总共提取了 {len(X_test)} 个样本的决策路径\n")
            f.write("="*100 + "\n")
        
        print(f"      Paths saved to: {filename}")
        return filename
    











