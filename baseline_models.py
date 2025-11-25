"""
基线模型模块 - Baseline Models Module
使用Optuna进行超参数优化的传统机器学习模型
包括: Logistic Regression, Logistic+L1, XGBoost, LightGBM, SVM, Random Forest
"""

import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

class BaselineModels:
    """基线模型类 - 使用Optuna优化超参数"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_models = {}
        self.best_params = {}
        self.results = {}
    
    def _evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
    
    def train_logistic(self, X_train, y_train, X_val, y_val, n_trials=50, penalty='l2'):
        """训练Logistic Regression（L2或L1正则化）"""
        
        def objective(trial):
            C = trial.suggest_float('C', 0.001, 100, log=True)
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            
            model = LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=1000,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        model = LogisticRegression(
            C=best_params['C'],
            penalty=penalty,
            solver=solver,
            max_iter=1000,
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        
        return model, best_params
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """训练XGBoost"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0.0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        best_params.update({
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        })
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return model, best_params
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """训练LightGBM"""
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        best_params.update({'random_state': self.random_state, 'verbose': -1})
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        return model, best_params
    
    def train_svm(self, X_train, y_train, X_val, y_val, n_trials=30):
        """训练SVM（较少的trials因为SVM较慢）"""
        
        def objective(trial):
            C = trial.suggest_float('C', 0.01, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            kernel = trial.suggest_categorical('kernel', ['rbf', 'linear'])
            
            model = SVC(
                C=C,
                gamma=gamma,
                kernel=kernel,
                random_state=self.random_state,
                max_iter=2000
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        model = SVC(
            C=best_params['C'],
            gamma=best_params['gamma'],
            kernel=best_params['kernel'],
            random_state=self.random_state,
            max_iter=2000
        )
        model.fit(X_train, y_train)
        
        return model, best_params
    
    def train_random_forest(self, X_train, y_train, X_val, y_val, n_trials=50):
        """训练Random Forest"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': self.random_state
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # 使用最优参数训练最终模型
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)
        
        return model, best_params
    
    def train_all_models(self, dataset_name, X_train, y_train, X_val, y_val, X_test, y_test):
        """训练所有基线模型"""
        print(f"\n{'='*80}")
        print(f"🔧 Training Baseline Models for {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # 检查并处理NaN值
        if np.isnan(X_train).any():
            print(f"⚠️  Warning: Found NaN in training data. Replacing with 0...")
            X_train = np.nan_to_num(X_train, nan=0.0)
        if np.isnan(X_val).any():
            print(f"⚠️  Warning: Found NaN in validation data. Replacing with 0...")
            X_val = np.nan_to_num(X_val, nan=0.0)
        if np.isnan(X_test).any():
            print(f"⚠️  Warning: Found NaN in test data. Replacing with 0...")
            X_test = np.nan_to_num(X_test, nan=0.0)
        
        # 检查无穷值
        if np.isinf(X_train).any():
            print(f"⚠️  Warning: Found Inf in training data. Replacing with large values...")
            X_train = np.nan_to_num(X_train, posinf=1e10, neginf=-1e10)
        if np.isinf(X_val).any():
            print(f"⚠️  Warning: Found Inf in validation data. Replacing with large values...")
            X_val = np.nan_to_num(X_val, posinf=1e10, neginf=-1e10)
        if np.isinf(X_test).any():
            print(f"⚠️  Warning: Found Inf in test data. Replacing with large values...")
            X_test = np.nan_to_num(X_test, posinf=1e10, neginf=-1e10)
        
        results = {}
        
        # 1. Logistic Regression (L2)
        print(f"  1/6 Logistic Regression (L2)...", end=' ')
        model, params = self.train_logistic(X_train, y_train, X_val, y_val, penalty='l2')
        metrics = self._evaluate_model(model, X_test, y_test)
        results['Logistic_L2'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 2. Logistic Regression (L1)
        print(f"  2/6 Logistic Regression (L1)...", end=' ')
        model, params = self.train_logistic(X_train, y_train, X_val, y_val, penalty='l1')
        metrics = self._evaluate_model(model, X_test, y_test)
        results['Logistic_L1'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 3. XGBoost
        print(f"  3/6 XGBoost...", end=' ')
        model, params = self.train_xgboost(X_train, y_train, X_val, y_val)
        metrics = self._evaluate_model(model, X_test, y_test)
        results['XGBoost'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 4. LightGBM
        print(f"  4/6 LightGBM...", end=' ')
        model, params = self.train_lightgbm(X_train, y_train, X_val, y_val)
        metrics = self._evaluate_model(model, X_test, y_test)
        results['LightGBM'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 5. SVM
        print(f"  5/6 SVM...", end=' ')
        model, params = self.train_svm(X_train, y_train, X_val, y_val)
        metrics = self._evaluate_model(model, X_test, y_test)
        results['SVM'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 6. Random Forest
        print(f"  6/6 Random Forest...", end=' ')
        model, params = self.train_random_forest(X_train, y_train, X_val, y_val)
        metrics = self._evaluate_model(model, X_test, y_test)
        results['RandomForest'] = {'model': model, 'params': params, 'metrics': metrics}
        print(f"✅ Acc: {metrics['accuracy']:.4f}")
        
        # 保存结果
        self.results[dataset_name] = results
        
        # 打印总结
        print(f"\n{'='*80}")
        print(f"📊 {dataset_name.upper()} - Baseline Models Summary:")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print(f"{'-'*80}")
        for model_name, result in results.items():
            m = result['metrics']
            print(f"{model_name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}")
        print(f"{'='*80}\n")
        
        return results


def train_all_baseline_models(processed_data):
    """为所有数据集训练基线模型"""
    print("\n" + "="*80)
    print("🚀 Phase: Baseline Models Training with Optuna Optimization")
    print("="*80)
    
    baseline_trainer = BaselineModels(random_state=42)
    all_results = {}
    
    for dataset_name, data in processed_data.items():
        results = baseline_trainer.train_all_models(
            dataset_name=dataset_name,
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            y_test=data['y_test']
        )
        all_results[dataset_name] = results
    
    # 保存结果到Excel
    save_baseline_results_to_excel(all_results)
    
    print("✅ All baseline models training completed!")
    return all_results, baseline_trainer


def save_baseline_results_to_excel(all_results):
    """将基线模型结果保存到Excel文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/baseline_models_comparison_{timestamp}.xlsx"
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    
    # 创建Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # 为每个数据集创建一个sheet
        for dataset_name, results in all_results.items():
            # 准备数据
            data = []
            for model_name, result in results.items():
                metrics = result['metrics']
                params = result['params']
                
                row = {
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'Parameters': str(params)
                }
                data.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 按准确率排序
            df = df.sort_values('Accuracy', ascending=False)
            
            # 写入Excel
            df.to_excel(writer, sheet_name=dataset_name.upper(), index=False)
        
        # 创建汇总sheet
        summary_data = []
        for dataset_name, results in all_results.items():
            for model_name, result in results.items():
                metrics = result['metrics']
                summary_data.append({
                    'Dataset': dataset_name.upper(),
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
        summary_df.to_excel(writer, sheet_name='SUMMARY', index=False)
    
    print(f"\n📊 Baseline models results saved to: {filename}")
    return filename


if __name__ == "__main__":
    # 测试代码
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_datasets()
    
    # 训练基线模型
    all_results, trainer = train_all_baseline_models(processed_data)
