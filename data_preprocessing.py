"""
数据预处理模块 - 更新版
Data Preprocessing Module - Updated Version
适配新的神经网络训练框架
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重现性
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self):
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = {}
    
    def load_german_credit(self):
        """加载German信用数据集"""
        print("🔄 Loading German Credit dataset...")
        
        # 从本地CSV文件加载
        try:
            df = pd.read_csv('data/german_credit.csv')
        except FileNotFoundError:
            print("❌ German credit data file not found. Please ensure 'data/german_credit.csv' exists.")
            return None
        
        print(f"German Credit original shape: {df.shape}")
        
        # 检查目标列
        if 'class' in df.columns:
            # 将目标变量转换为二进制（1=良好信用，0=不良信用）
            df['class'] = df['class'].replace({1: 1, 2: 0})  # 1=good, 2=bad -> 1=good, 0=bad
            target_col = 'class'
        elif 'Class' in df.columns:
            df['Class'] = df['Class'].replace({1: 1, 2: 0})
            target_col = 'Class'
        else:
            # 假设最后一列是目标列
            target_col = df.columns[-1]
            df[target_col] = df[target_col].replace({1: 1, 2: 0})
        
        # 识别分类变量和数值变量
        categorical_cols = []
        numerical_cols = []
        
        for col in df.columns:
            if col != target_col:
                if df[col].dtype == 'object' or df[col].nunique() <= 10:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
        
        # 处理分类变量 - One-hot编码
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # 分割特征和目标
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        print(f"German Credit processed shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # 分割数据集：60% 训练，20% 验证，20% 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 x 0.8 = 0.2 of total
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存scaler和特征名
        self.scalers['german'] = scaler
        self.feature_names['german'] = list(X.columns)
        
        print(f"German dataset split: Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
    
    def load_xinwang(self):
        """加载Xinwang信用数据集"""
        print("🔄 Loading Xinwang Credit dataset...")
        
        try:
            df = pd.read_csv('data/xinwang.csv')
        except FileNotFoundError:
            print("❌ Xinwang credit data file not found. Please ensure 'data/xinwang.csv' exists.")
            return None
        
        print(f"Xinwang Credit original shape: {df.shape}")
        
        # 目标列是'target'
        target_col = 'target'
        
        # 处理缺失值 - 用-99表示的缺失值
        for col in df.columns:
            if col != target_col:
                # 将-99替换为NaN
                df[col] = df[col].replace(-99, np.nan)
                # 用中位数填充数值型缺失值
                if df[col].dtype != 'object':
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        # 处理可能的分类变量 (province, industry, scope, judicial)
        categorical_cols = ['province', 'industry', 'scope', 'judicial']
        for col in categorical_cols:
            if col in df.columns and df[col].dtype == 'object':
                # 使用Label Encoding处理分类变量
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[f'xinwang_{col}'] = le
        
        # 分离特征和目标
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 确保目标变量是0和1
        if y.min() != 0 or y.max() != 1:
            print(f"⚠️  Warning: Target values range from {y.min()} to {y.max()}, mapping to 0-1")
            y = (y > 0).astype(int)
        
        # 数据集划分：60% train, 20% validation, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存scaler和特征名
        self.scalers['xinwang'] = scaler
        self.feature_names['xinwang'] = list(X.columns)
        
        print(f"Xinwang dataset split: Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"Target distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
    
    def load_australian_credit(self):
        """加载Australian信用数据集"""
        print("🔄 Loading Australian Credit dataset...")
        
        try:
            df = pd.read_csv('data/australian_credit.csv')
        except FileNotFoundError:
            print("❌ Australian credit data file not found. Please ensure 'data/australian_credit.csv' exists.")
            return None
        
        print(f"Australian Credit original shape: {df.shape}")
        
        # 检查目标列
        if 'Class' in df.columns:
            target_col = 'Class'
        elif 'class' in df.columns:
            target_col = 'class'
        else:
            # 假设最后一列是目标列
            target_col = df.columns[-1]
        
        # 确保目标变量是0和1
        unique_values = df[target_col].unique()
        if len(unique_values) == 2:
            # 将目标变量映射为0和1
            value_mapping = {unique_values[0]: 0, unique_values[1]: 1}
            df[target_col] = df[target_col].map(value_mapping)
        
        # 处理缺失值
        for col in df.columns:
            if col != target_col:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        
        # 识别并处理分类变量
        categorical_cols = []
        for col in df.columns:
            if col != target_col and (df[col].dtype == 'object' or df[col].nunique() <= 10):
                categorical_cols.append(col)
        
        # One-hot编码
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # 分割特征和目标
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        print(f"Australian Credit processed shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # 分割数据集：60% 训练，20% 验证，20% 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存scaler和特征名
        self.scalers['australian'] = scaler
        self.feature_names['australian'] = list(X.columns)
        
        print(f"Australian dataset split: Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
    
    def load_uci_credit(self):
        """加载UCI信用数据集"""
        print("🔄 Loading UCI Credit dataset...")
        
        try:
            # 尝试读取Excel文件
            df = pd.read_excel('data/uci_credit.xls', header=1, index_col=0)
        except FileNotFoundError:
            print("❌ UCI credit data file not found. Please ensure 'data/uci_credit.xls' exists.")
            return None
        except Exception as e:
            print(f"❌ Error loading UCI credit data: {e}")
            return None
        
        print(f"UCI Credit original shape: {df.shape}")
        
        # 重命名目标列
        if 'default payment next month' in df.columns:
            df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
            target_col = 'DEFAULT'
        elif 'DEFAULT' in df.columns:
            target_col = 'DEFAULT'
        else:
            # 假设最后一列是目标列
            target_col = df.columns[-1]
        
        # 移除ID列如果存在
        if 'ID' in df.columns:
            df.drop('ID', axis=1, inplace=True)
        
        # 处理性别编码异常值
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].replace({0: 2})  # 0替换为2，确保只有1和2
        
        # 处理教育和婚姻状况的异常值
        if 'EDUCATION' in df.columns:
            df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # 合并未知类别
        
        if 'MARRIAGE' in df.columns:
            df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})  # 0替换为3
        
        # 强制所有特征为数值型
        for col in df.columns:
            if col != target_col:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
        
        # 分割特征和目标
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        print(f"UCI Credit processed shape: {X.shape}")
        print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # 分割数据集：60% 训练，20% 验证，20% 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 保存scaler和特征名
        self.scalers['uci'] = scaler
        self.feature_names['uci'] = list(X.columns)
        
        print(f"UCI dataset split: Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
    
    def split_and_scale_data(self, X, y, feature_names, test_size=0.2, val_size=0.2):
        """分割和标准化数据"""
        # 首先分割训练集和测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 再从训练集中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
    
    def process_all_datasets(self):
        """处理所有数据集"""
        print("📊 Processing all datasets...")
        
        processed_data = {}
        
        # 处理German数据集
        german_data = self.load_german_credit()
        if german_data is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = german_data
            processed_data['german'] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': self.feature_names['german'],
                'scaler': self.scalers['german']
            }
            print(f"🔧 Processing german dataset...")
            print(f"german dataset split:")
            print(f"  Train: {processed_data['german']['X_train'].shape}, Val: {processed_data['german']['X_val'].shape}, Test: {processed_data['german']['X_test'].shape}")
            print(f"  Class distribution - Train: {np.bincount(processed_data['german']['y_train'])}, Val: {np.bincount(processed_data['german']['y_val'])}, Test: {np.bincount(processed_data['german']['y_test'])}")
        
        # 处理Australian数据集
        australian_data = self.load_australian_credit()
        if australian_data is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = australian_data
            processed_data['australian'] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': self.feature_names['australian'],
                'scaler': self.scalers['australian']
            }
            print(f"🔧 Processing australian dataset...")
            print(f"australian dataset split:")
            print(f"  Train: {processed_data['australian']['X_train'].shape}, Val: {processed_data['australian']['X_val'].shape}, Test: {processed_data['australian']['X_test'].shape}")
            print(f"  Class distribution - Train: {np.bincount(processed_data['australian']['y_train'])}, Val: {np.bincount(processed_data['australian']['y_val'])}, Test: {np.bincount(processed_data['australian']['y_test'])}")
        
        # 处理UCI数据集
        uci_data = self.load_uci_credit()
        if uci_data is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = uci_data
            processed_data['uci'] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': self.feature_names['uci'],
                'scaler': self.scalers['uci']
            }
            print(f"🔧 Processing uci dataset...")
            print(f"uci dataset split:")
            print(f"  Train: {processed_data['uci']['X_train'].shape}, Val: {processed_data['uci']['X_val'].shape}, Test: {processed_data['uci']['X_test'].shape}")
            print(f"  Class distribution - Train: {np.bincount(processed_data['uci']['y_train'])}, Val: {np.bincount(processed_data['uci']['y_val'])}, Test: {np.bincount(processed_data['uci']['y_test'])}")
        
        # 处理Xinwang数据集
        xinwang_data = self.load_xinwang()
        if xinwang_data is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = xinwang_data
            processed_data['xinwang'] = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'feature_names': self.feature_names['xinwang'],
                'scaler': self.scalers['xinwang']
            }
            print(f"🔧 Processing xinwang dataset...")
            print(f"xinwang dataset split:")
            print(f"  Train: {processed_data['xinwang']['X_train'].shape}, Val: {processed_data['xinwang']['X_val'].shape}, Test: {processed_data['xinwang']['X_test'].shape}")
            print(f"  Class distribution - Train: {np.bincount(processed_data['xinwang']['y_train'])}, Val: {np.bincount(processed_data['xinwang']['y_val'])}, Test: {np.bincount(processed_data['xinwang']['y_test'])}")
        
        print("✅ All datasets processed successfully!")
        return processed_data

if __name__ == "__main__":
    # 测试数据预处理
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_all_datasets()
    
    for dataset_name, data in processed_data.items():
        print(f"\n{dataset_name.upper()} Dataset Summary:")
        print(f"  Features: {len(data['feature_names'])}")
        print(f"  Train samples: {data['X_train'].shape[0]}")
        print(f"  Validation samples: {data['X_val'].shape[0]}")
        print(f"  Test samples: {data['X_test'].shape[0]}")
        print(f"  Class distribution (train): {np.bincount(data['y_train'])}")
