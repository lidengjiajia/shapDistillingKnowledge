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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os
from datetime import datetime

# 并发配置：使用CPU核心数-1，至少为1
n_jobs = max(1, min(mp.cpu_count() - 1, mp.cpu_count()))
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
                left_rule = f"{parent_rule}{name} <= {threshold:.3f}"
                right_rule = f"{parent_rule}{name} > {threshold:.3f}"
                recurse(tree.children_left[node], depth + 1, left_rule + " and ")
                recurse(tree.children_right[node], depth + 1, right_rule + " and ")
            else:
                # 叶子节点
                if parent_rule:
                    rule = parent_rule.rstrip(" and ")
                    value = tree.value[node]
                    predicted_class = np.argmax(value)
                    confidence = np.max(value) / np.sum(value)
                    rules.append(f"IF {rule} THEN class={predicted_class} (confidence={confidence:.3f})")
        
        try:
            recurse(0, 0)
        except Exception as e:
            # 如果规则提取失败，返回简单描述
            rules = [f"Decision tree with {tree.node_count} nodes"]
        
        return rules
    
    def train_baseline_decision_tree(self, dataset_name):
        """训练基础决策树（不使用蒸馏）"""
        data_dict = self.processed_data[dataset_name]
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        feature_names = data_dict['feature_names']
        
        # 固定参数训练基础决策树（无Optuna）
        model = DecisionTreeClassifier(
            max_depth=5,
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
                            'Best': f"{best_accuracy:.3f}"
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
            
            # 设置并发数量
            import platform
            if platform.system() == 'Windows':
                n_jobs = min(4, max(1, mp.cpu_count() // 2))
            else:
                n_jobs = max(1, min(mp.cpu_count() - 1, mp.cpu_count()))
            
            print(f"     🚀 Using {n_jobs} parallel jobs for Top-k distillation")
            
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
            
            # 使用进程池并行执行
            if platform.system() == 'Windows':
                # Windows下使用spawn方法避免pickle问题
                mp.set_start_method('spawn', force=True)
            
            from multiprocessing import Pool
            from functools import partial
            
            # 由于需要访问self，我们需要特殊处理
            # 序列化实验函数
            def experiment_worker(params):
                dataset_name, k, temperature, alpha, max_depth = params
                # 重新创建所需的对象（这是并发的代价）
                # 实际执行将在主进程中完成，这里改为串行但有进度显示
                return params
            
            # 因为self无法序列化，改为使用线程池来实现并发
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
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
                        'Best': f"{best_accuracy:.3f}"
                    })
                    progress_bar.update(1)
            
            progress_bar.close()
            results[dataset_name]['best'] = best_result
            results[dataset_name]['best_k'] = best_k
            print(f"     Best Accuracy: {best_accuracy:.4f} with k={best_k}")  # 改为显示准确率
        
        # 保存Top-k消融实验数据和创建可视化
        print("\n📊 Saving Top-k ablation study data and creating visualizations...")
        topk_ablation_analyzer.save_ablation_data(prefix='topk_ablation_study')
        # 使用通用的消融实验可视化方法（避免重复生成）
        topk_ablation_analyzer.create_ablation_visualizations()
        topk_ablation_analyzer.generate_summary_report(prefix='topk_ablation_study')
        
        return results
    


