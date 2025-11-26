"""
SHAP特征重要性分析模块
SHAP Feature Importance Analysis Module
"""

import numpy as np
import os
# 设置matplotlib后端为非交互式，避免多线程问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import shap
import warnings
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# 禁用Optuna日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('default')
# 设置默认字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class SHAPAnalyzer:
    """SHAP特征重要性分析器 - 基于决策树模型"""
    
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.decision_tree_models = {}
        
        # 设置并发数量：Windows平台强制使用单线程避免中文路径编码问题
        import platform
        if platform.system() == 'Windows':
            # Windows下禁用并行，避免joblib/multiprocessing的中文路径编码问题
            self.n_jobs = 1
            print(f"🔧 SHAP Analyzer initialized with n_jobs=1 (Windows - avoiding encoding issues)")
        else:
            # Linux/Mac可以使用更多并发
            self.n_jobs = max(1, min(os.cpu_count() - 1, os.cpu_count()))
            print(f"🔧 SHAP Analyzer initialized with {self.n_jobs} parallel jobs (CPU cores: {os.cpu_count()}, Platform: {platform.system()})")
        
    def train_decision_trees(self):
        """Train decision tree models for each dataset for SHAP analysis"""
        print("🌳 Training decision trees for SHAP analysis...")
        
        from tqdm import tqdm
        datasets = list(self.processed_data.items())
        
        for dataset_name, data_dict in tqdm(datasets, desc="🌳 Training Trees", unit="dataset"):
            print(f"   Training decision tree for {dataset_name}...")
            
            X_train = data_dict['X_train']
            X_test = data_dict['X_test']
            y_train = data_dict['y_train']
            y_test = data_dict['y_test']
            
            # Use Optuna to optimize decision tree parameters
            def objective(trial):
                # Define hyperparameter search space
                max_depth = trial.suggest_int('max_depth', 5, 25)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                
                # Create decision tree model
                dt = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                # Evaluate model using cross-validation with parallel processing
                scores = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy', n_jobs=self.n_jobs)
                return scores.mean()
            
            # Create Optuna study and optimize with parallel execution
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=50, show_progress_bar=False, n_jobs=self.n_jobs)
            
            # Train final model with best parameters
            best_params = study.best_params
            best_model = DecisionTreeClassifier(
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            
            # Calculate test set accuracy
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.decision_tree_models[dataset_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'best_params': best_params,
                'best_score': study.best_value
            }
            
            print(f"     Decision tree accuracy: {accuracy:.4f}")
            print(f"     Best params: {best_params}")
            print(f"     CV score: {study.best_value:.4f}")
        
        print("✅ Decision trees trained for SHAP analysis")
        
    def compute_shap_values(self, dataset_name, top_k_range=None):
        """Compute SHAP values using decision tree model with full dataset
        
        SHAP computation methodology:
        1. Use ALL available data (train + validation + test) for accurate SHAP computation
        2. Use SHAP TreeExplainer optimized for tree models
        3. Calculate SHAP values for each individual sample
        4. Aggregate feature importance through mean absolute SHAP values
        5. Ensure precise feature ranking without duplicates
        
        Args:
            dataset_name: 数据集名称
            top_k_range: k值范围，如果为None则自动设置为(5, 特征总数)
        """
        # 如果没有指定k范围，则根据数据集特征数量自动设置
        if top_k_range is None:
            data_dict = self.processed_data[dataset_name]
            n_features = len(data_dict['feature_names'])
            top_k_range = (5, n_features)
        
        print(f"\n🔍 Computing SHAP values for {dataset_name.upper()} dataset...")
        print(f"   Method: TreeExplainer with decision tree model")
        print(f"   Using FULL dataset for accurate SHAP computation")
        print(f"   Top-k range: {top_k_range[0]} to {top_k_range[1]}")
        
        model_info = self.decision_tree_models[dataset_name]
        model = model_info['model']
        data_dict = self.processed_data[dataset_name]
        
        # Use ALL available data: train + validation + test
        X_train = data_dict['X_train']
        X_val = data_dict['X_val'] 
        X_test = data_dict['X_test']
        
        # Combine all data for comprehensive SHAP analysis
        import numpy as np
        X_all = np.vstack([X_train, X_val, X_test])
        
        print(f"   Data samples: {X_train.shape[0]} train + {X_val.shape[0]} val + {X_test.shape[0]} test = {X_all.shape[0]} total")
        print(f"   Feature dimensions: {X_all.shape[1]} features")
        
        # Create SHAP TreeExplainer with model check model
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        except:
            explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values with proper error handling
        print(f"   Calculating SHAP values for {X_all.shape[0]} samples...")
        try:
            shap_values = explainer.shap_values(X_all, check_additivity=False)
        except:
            shap_values = explainer.shap_values(X_all)
        
        # Handle different SHAP output formats carefully
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                # Binary classification - use positive class SHAP values
                shap_values = shap_values[1]
                print(f"   Using positive class SHAP values (binary classification)")
            else:
                shap_values = shap_values[0]
                print(f"   Using first class SHAP values")
        
        print(f"   SHAP values shape: {shap_values.shape}")
        
        # Calculate feature importance with proper validation
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Ensure feature_importance is properly formatted and has correct dimensions
        if feature_importance.ndim > 1:
            feature_importance = feature_importance.flatten()
        feature_importance = feature_importance.astype(float)
        
        # Debug: Check dimensions
        feature_names = data_dict['feature_names']
        print(f"   Feature names count: {len(feature_names)}")
        print(f"   Feature importance count: {len(feature_importance)}")
        print(f"   X_all shape: {X_all.shape}")
        
        # Fix dimension mismatch if it exists
        if len(feature_importance) != len(feature_names):
            if len(feature_importance) == 2 * len(feature_names):
                # Likely duplicated due to binary classification - take first half
                feature_importance = feature_importance[:len(feature_names)]
                print(f"   Fixed dimension mismatch: took first {len(feature_names)} values")
            elif len(feature_names) != X_all.shape[1]:
                # Feature names and X_all dimensions don't match
                print(f"   Warning: Feature names ({len(feature_names)}) != X_all features ({X_all.shape[1]})")
                # Use actual X_all dimensions
                if len(feature_importance) == X_all.shape[1]:
                    # Create generic feature names
                    feature_names = [f'feature_{i}' for i in range(X_all.shape[1])]
                    print(f"   Created generic feature names: {len(feature_names)} features")
                else:
                    raise ValueError(f"Cannot resolve dimension mismatch: importance({len(feature_importance)}) vs features({X_all.shape[1]})")
            else:
                raise ValueError(f"Cannot resolve dimension mismatch: names({len(feature_names)}) vs importance({len(feature_importance)})")
        
        # Validate feature importance calculation
        print(f"   Final feature importance shape: {feature_importance.shape}")
        print(f"   Feature importance range: [{feature_importance.min():.6f}, {feature_importance.max():.6f}]")
        
        # Create feature importance dictionary with corrected dimensions
        importance_dict = dict(zip(feature_names, feature_importance))
        
        # Sort by importance with proper handling
        sorted_features = sorted(importance_dict.items(), key=lambda x: float(x[1]), reverse=True)
        
        print(f"   Top 8 important features for {dataset_name}:")
        for i, (feature, importance) in enumerate(sorted_features[:8]):
            print(f"     {i+1}. {feature}: {float(importance):.6f}")
        
        # Verify no duplicate importance values in top features
        top_importances = [float(x[1]) for x in sorted_features[:8]]
        unique_importances = len(set(top_importances))
        print(f"   Unique importance values in top 8: {unique_importances}/8")
        
        # Generate different top-k feature selections
        top_k_features = {}
        for k in range(top_k_range[0], top_k_range[1] + 1):
            top_k_features[k] = [feat[0] for feat in sorted_features[:k]]
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'sorted_features': sorted_features,
            'top_k_features': top_k_features,
            'explainer': explainer,
            'feature_names': feature_names
        }
    
    def create_combined_shap_visualization(self, all_shap_results):
        """Create separate SHAP visualizations for three datasets with top 10 features"""
        print(f"📊 Creating separate SHAP visualizations with top 10 features...")
        
        # 按要求的顺序：German, Australian, UCI
        datasets = ['german', 'australian', 'uci']
        titles = ['German Credit Dataset', 'Australian Credit Dataset', 'UCI Credit Dataset']
        filenames = ['shap_german_features.png', 'shap_australian_features.png', 'shap_uci_features.png']
        
        saved_files = []
        
        for idx, (dataset_name, title, filename) in enumerate(zip(datasets, titles, filenames)):
            # 创建单独的图形
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            shap_results = all_shap_results[dataset_name]
            
            # 获取Top 10特征 - 使用真实特征名
            top_features = shap_results['sorted_features'][:10]
            features, importances = zip(*top_features)
            importances = [float(x) for x in importances]
            
            # 获取真实的原始特征名
            real_feature_names = self._get_real_feature_names(dataset_name, features)
            
            # 创建统一色调的配色方案 - 使用柔和的颜色
            if idx == 0:  # German - 柔和蓝色
                base_colors = ['#7BB3F0'] * 10  # 使用同一个柔和颜色
            elif idx == 1:  # Australian - 柔和紫色
                base_colors = ['#DDA0DD'] * 10  # 使用同一个柔和颜色
            else:  # UCI - 柔和橙色
                base_colors = ['#FFB366'] * 10  # 使用同一个柔和颜色
            
            # 创建条形图 - 改进视觉效果
            bars = ax.barh(range(len(real_feature_names)), importances, 
                          color=base_colors, alpha=0.9, edgecolor='white', linewidth=1.5)
            
            ax.set_yticks(range(len(real_feature_names)))
            ax.set_yticklabels(real_feature_names, fontsize=13, fontweight='normal')
            ax.set_xlabel('Mean |SHAP Value|', fontsize=15, fontweight='bold')
            ax.set_title(title, fontsize=18, fontweight='bold', pad=25)
            ax.invert_yaxis()
            
            # 添加网格线以提高可读性
            ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # 添加数值标签 - 改进格式
            max_imp = max(importances)
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max_imp*0.01, 
                       bar.get_y() + bar.get_height()/2, 
                       f'{imp:.4f}', va='center', fontsize=11, 
                       fontweight='bold', color='black')
            
            # 美化坐标轴
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['bottom'].set_color('#CCCCCC')
            
            # 设置背景色
            ax.set_facecolor('#FAFAFA')
            
            plt.tight_layout(pad=3.0)
            filepath = f'results/{filename}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            saved_files.append(filepath)
            print(f"   ✅ {title} SHAP visualization saved to: {filepath}")
        
        print(f"   ✅ All SHAP feature importance visualizations completed")
        
        return saved_files
    
    def _get_real_feature_names(self, dataset_name, encoded_features):
        """Get real feature names from original datasets"""
        real_names = []
        
        for feature in encoded_features:
            if dataset_name == 'german':
                # German dataset - map to meaningful English names
                if 'Status_A12' in feature:
                    real_names.append('Account Status (A12)')
                elif 'Status_A13' in feature:
                    real_names.append('Account Status (A13)')
                elif 'Status_A14' in feature:
                    real_names.append('Account Status (A14)')
                elif 'Purpose_A410' in feature:
                    real_names.append('Purpose (New Car)')
                elif 'Purpose_A41' in feature:
                    real_names.append('Purpose (Used Car)')
                elif 'Purpose_A42' in feature:
                    real_names.append('Purpose (Furniture)')
                elif 'Purpose_A43' in feature:
                    real_names.append('Purpose (Radio/TV)')
                elif 'Duration' in feature:
                    real_names.append('Credit Duration')
                elif 'Credit_amount' in feature:
                    real_names.append('Credit Amount')
                elif 'Age' in feature:
                    real_names.append('Age')
                elif 'Savings_A61' in feature:
                    real_names.append('Savings (<100 DM)')
                elif 'Savings_A62' in feature:
                    real_names.append('Savings (100-500 DM)')
                elif 'Employment_A71' in feature:
                    real_names.append('Employment (Unemployed)')
                elif 'Employment_A72' in feature:
                    real_names.append('Employment (<1 year)')
                else:
                    real_names.append(feature)
                    
            elif dataset_name == 'uci':
                # UCI Taiwan dataset - map to meaningful English names
                if feature == 'PAY_0':
                    real_names.append('Payment Status (Sep)')
                elif feature == 'PAY_2':
                    real_names.append('Payment Status (Aug)')
                elif feature == 'PAY_3':
                    real_names.append('Payment Status (Jul)')
                elif feature == 'PAY_4':
                    real_names.append('Payment Status (Jun)')
                elif feature == 'PAY_5':
                    real_names.append('Payment Status (May)')
                elif feature == 'PAY_6':
                    real_names.append('Payment Status (Apr)')
                elif feature == 'BILL_AMT1':
                    real_names.append('Bill Amount (Sep)')
                elif feature == 'BILL_AMT2':
                    real_names.append('Bill Amount (Aug)')
                elif feature == 'BILL_AMT3':
                    real_names.append('Bill Amount (Jul)')
                elif feature == 'BILL_AMT4':
                    real_names.append('Bill Amount (Jun)')
                elif feature == 'BILL_AMT5':
                    real_names.append('Bill Amount (May)')
                elif feature == 'BILL_AMT6':
                    real_names.append('Bill Amount (Apr)')
                elif feature == 'PAY_AMT1':
                    real_names.append('Payment Amount (Sep)')
                elif feature == 'PAY_AMT2':
                    real_names.append('Payment Amount (Aug)')
                elif feature == 'LIMIT_BAL':
                    real_names.append('Credit Limit')
                elif feature == 'SEX':
                    real_names.append('Gender')
                elif feature == 'EDUCATION':
                    real_names.append('Education Level')
                elif feature == 'MARRIAGE':
                    real_names.append('Marital Status')
                elif feature == 'AGE':
                    real_names.append('Age')
                else:
                    real_names.append(feature)
                    
            elif dataset_name == 'australian':
                # Australian dataset - features are anonymous, use generic names
                if feature == 'feature_1':
                    real_names.append('Feature 1 (Continuous)')
                elif feature == 'feature_2':
                    real_names.append('Feature 2 (Continuous)')
                elif feature == 'feature_4':
                    real_names.append('Feature 4 (Continuous)')
                elif feature == 'feature_6':
                    real_names.append('Feature 6 (Continuous)')
                elif feature == 'feature_9':
                    real_names.append('Feature 9 (Binary)')
                elif feature == 'feature_12':
                    real_names.append('Feature 12 (Binary)')
                elif feature == 'feature_13':
                    real_names.append('Feature 13 (Continuous)')
                elif 'feature_0_1' in feature:
                    real_names.append('Feature 0 (Category 1)')
                elif 'feature_3_2' in feature:
                    real_names.append('Feature 3 (Category 2)')
                elif 'feature_3_3' in feature:
                    real_names.append('Feature 3 (Category 3)')
                elif 'feature_5_2' in feature:
                    real_names.append('Feature 5 (Category 2)')
                elif 'feature_5_3' in feature:
                    real_names.append('Feature 5 (Category 3)')
                elif 'feature_5_4' in feature:
                    real_names.append('Feature 5 (Category 4)')
                elif 'feature_5_5' in feature:
                    real_names.append('Feature 5 (Category 5)')
                else:
                    real_names.append(feature.replace('feature_', 'Feature '))
            else:
                real_names.append(feature)
        
        return real_names
    
    def _get_original_feature_names(self, dataset_name, encoded_features):
        """Convert encoded feature names back to original names when possible"""
        original_names = []
        
        for feature in encoded_features:
            if dataset_name == 'german':
                # German dataset original feature mappings - 保持具体特征名以避免重复
                if 'Status_A1' in feature:
                    original_names.append(f'Account Status ({feature})')
                elif 'Duration' in feature:
                    original_names.append('Duration')
                elif 'Credit_amount' in feature:
                    original_names.append('Credit Amount')
                elif 'Purpose_A4' in feature:
                    # 保持具体的Purpose编码以避免重复
                    purpose_code = feature.replace('Purpose_', '')
                    original_names.append(f'Purpose ({purpose_code})')
                elif 'Age' in feature:
                    original_names.append('Age')
                elif 'Savings_A6' in feature:
                    original_names.append(f'Savings ({feature})')
                elif 'Employment_A7' in feature:
                    original_names.append(f'Employment ({feature})')
                elif 'Property_A12' in feature:
                    original_names.append(f'Property ({feature})')
                else:
                    original_names.append(feature)
            elif dataset_name == 'uci':
                # UCI Taiwan dataset original feature mappings - 保持具体性
                if 'PAY_' in feature:
                    pay_month = feature.split('_')[1] if '_' in feature else '?'
                    original_names.append(f'Payment Status M{pay_month}')
                elif 'BILL_AMT' in feature:
                    bill_month = feature.split('T')[1] if 'T' in feature else '?'
                    original_names.append(f'Bill Amount M{bill_month}')
                elif 'PAY_AMT' in feature:
                    pay_month = feature.split('T')[1] if 'T' in feature else '?'
                    original_names.append(f'Payment Amount M{pay_month}')
                elif 'LIMIT_BAL' in feature:
                    original_names.append('Credit Limit')
                elif 'SEX' in feature:
                    original_names.append('Gender')
                elif 'EDUCATION' in feature:
                    original_names.append('Education')
                elif 'MARRIAGE' in feature:
                    original_names.append('Marriage')
                elif 'AGE' in feature:
                    original_names.append('Age')
                else:
                    original_names.append(feature)
            elif dataset_name == 'australian':
                # Australian dataset - features are anonymous, use complete feature names
                if 'feature_' in feature:
                    # 保持完整的特征名，避免重复
                    original_names.append(f'Feature {feature.replace("feature_", "")}')
                else:
                    original_names.append(feature)
            else:
                    original_names.append(feature)
        
        return original_names
    
    def visualize_shap_importance(self, dataset_name, shap_results, save_path=None):
        """可视化SHAP特征重要性 - 改进配色和高分辨率
        
        Args:
            dataset_name: 数据集名称
            shap_results: compute_shap_values返回的结果字典
            save_path: 保存路径，如果为None则不保存
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 获取特征重要性
        sorted_features = shap_results['sorted_features']
        feature_names = [f[0] for f in sorted_features[:10]]  # 只取前10个特征
        importances = [f[1] for f in sorted_features[:10]]
        
        # 创建高分辨率图表
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        
        # 使用渐变色彩方案 - 从深蓝到浅蓝
        colors = plt.cm.RdYlBu_r(np.linspace(0.3, 0.8, len(feature_names)))
        
        # 绘制水平条形图
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importances, color=colors, edgecolor='#2C3E50', linewidth=1.2, alpha=0.85)
        
        # 在条形图上添加数值标签
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax.text(val + max(importances)*0.01, i, f'{val:.4f}', 
                   va='center', fontsize=10, fontweight='bold', color='#2C3E50')
        
        # 设置标签 - 更大更清晰的字体
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=11, fontweight='600')
        ax.invert_yaxis()  # 最重要的特征在顶部
        ax.set_xlabel('Mean |SHAP Value|', fontsize=13, fontweight='bold', color='#2C3E50')
        ax.set_title(f'SHAP Feature Importance - {dataset_name.upper()}', 
                    fontsize=16, fontweight='bold', color='#1A252F', pad=20)
        
        # 美化网格
        ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8, color='#7F8C8D')
        ax.set_axisbelow(True)
        
        # 设置背景色
        ax.set_facecolor('#F8F9FA')
        fig.patch.set_facecolor('white')
        
        # 添加边框
        for spine in ax.spines.values():
            spine.set_edgecolor('#2C3E50')
            spine.set_linewidth(1.5)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存高分辨率图片
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white', 
                       edgecolor='none', format='png', metadata={'Software': 'SHAP Analysis'})
            print(f"   📊 SHAP visualization saved (600 DPI): {save_path}")
        
        plt.close()

