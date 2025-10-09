# 🎯 基于SHAP引导的知识蒸馏信用评分系统# SHAP-Guided Knowledge Distillation for Credit Scoring



## 📋 项目简介## 🎯 Project Overview



**SHAP-Guided Knowledge Distillation for Credit Scoring****基于SHAP特征重要性引导的知识蒸馏信用评分系统**



本项目实现了一个创新的信用评分系统，将深度神经网络的预测能力与决策树的可解释性相结合，通过知识蒸馏技术实现智能特征选择和模型压缩。系统使用SHAP（SHapley Additive exPlanations）方法进行特征重要性分析，引导知识蒸馏过程，在保持高准确率的同时大幅提升模型可解释性。This project implements a comprehensive framework for **SHAP-guided knowledge distillation** in credit scoring applications. The system combines the interpretability of decision trees with the predictive power of deep neural networks through innovative knowledge distillation techniques, using SHAP (SHapley Additive exPlanations) for intelligent feature selection.



### ✨ 核心创新点---



1. **SHAP引导的特征选择**: 使用SHAP值识别最重要的特征，实现智能降维## 📁 Project Structure

2. **知识蒸馏技术**: 将复杂神经网络（教师模型）的知识迁移到简单决策树（学生模型）

3. **全面的消融实验**: 系统分析Top-k特征数量、温度参数、蒸馏权重等对模型性能的影响```

4. **可解释决策规则**: 自动提取并展示易于理解的决策规则Financial innovation/

├── data/                          # Dataset storage

---│   ├── german_credit.csv         # German Credit Dataset (1,000 samples, 54 features)

│   ├── australian_credit.csv     # Australian Credit Dataset (690 samples, 22 features)

## 📁 项目结构│   └── uci_credit.xls           # UCI Taiwan Credit Dataset (30,000 samples, 23 features)

├── results/                       # Generated output files

```│   ├── model_comparison_*.xlsx    # Model performance comparison

Financial innovation/│   ├── shap_feature_importance.png # SHAP feature visualization

├── data/                              # 数据集目录│   ├── ablation_study_analysis_*.png # Ablation study plots

│   ├── german_credit.csv             # 德国信用数据集 (1,000样本, 54特征)│   ├── topk_ablation_study_analysis_*.png # Top-k ablation analysis

│   ├── australian_credit.csv         # 澳大利亚信用数据集 (690样本, 22特征)│   ├── best_all_feature_rules_*.txt # Full feature decision rules

│   └── uci_credit.xls               # UCI台湾信用数据集 (30,000样本, 23特征)│   └── best_topk_rules_*.txt     # Top-k feature decision rules

├── results/                           # 实验结果目录├── main.py                       # Main execution pipeline

│   ├── model_comparison_*.xlsx        # 模型性能对比表├── data_preprocessing.py         # Data loading and preprocessing

│   ├── shap_*_features.png           # SHAP特征重要性可视化├── neural_models.py             # Neural network teacher models

│   ├── topk_ablation_visualization_*.png  # Top-k消融实验分析图├── distillation_module.py       # Knowledge distillation core

│   ├── depth_ablation_visualization_*.png # 决策树深度消融分析图├── shap_analysis.py             # SHAP feature importance analysis

│   ├── best_all_feature_rules_*.txt   # 全特征决策规则├── ablation_analyzer.py         # Ablation study visualization

│   └── best_topk_rules_*.txt         # Top-k特征决策规则├── result_manager.py            # Output management and reporting

├── main.py                           # 主程序入口└── README.md                    # This documentation

├── data_preprocessing.py             # 数据加载和预处理```

├── neural_models.py                  # 神经网络教师模型

├── distillation_module.py           # 知识蒸馏核心模块---

├── shap_analysis.py                 # SHAP特征重要性分析

├── ablation_analyzer.py             # 消融实验可视化## 🧠 Teacher Model Architectures

├── result_manager.py                # 结果管理和报告生成

├── clean_results.py                 # 结果清理脚本### German Credit Dataset (1,000 samples, 54 features)

└── README.md                        # 项目文档**Enhanced Residual Neural Network** - 优化的残差网络架构

```- **Architecture**: Residual blocks with skip connections for improved gradient flow

- **Layers**:

---  - Input: Linear(54 → 512) + BatchNorm + ReLU + Dropout(0.3)

  - Residual Block 1: [Linear(512 → 256) → BatchNorm → ReLU → Linear(256 → 256) → BatchNorm] + Skip(512 → 256)

## 🔬 技术架构  - Residual Block 2: [Linear(256 → 128) → BatchNorm → ReLU → Linear(128 → 128) → BatchNorm] + Skip(256 → 128)

  - Output: Linear(128 → 64) → BatchNorm → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)

### 1. 教师模型（Teacher Models）- **Loss Function**: BCEWithLogitsLoss with class balancing (pos_weight for imbalanced data)

- **Optimization**: AdamW (lr=0.0005, weight_decay=1e-3), ReduceLROnPlateau scheduler

#### German信用数据集- **Training**: 100 epochs (optimized), patience=30, batch_size=32

- **架构**: 增强残差神经网络 (Enhanced Residual Network)- **Target Accuracy**: 75%+ (improved from previous 62%)

- **特点**: 残差连接 + 批归一化 + Dropout正则化- **Reference**: Residual Networks (ResNet) - He et al. (2016)

- **层结构**:

  - 输入层: Linear(54 → 512) + BatchNorm + ReLU + Dropout(0.3)### Australian Credit Dataset (690 samples, 22 features)  

  - 残差块1: [Linear(512 → 256) → BatchNorm → ReLU → Linear(256 → 256) → BatchNorm] + Skip(512 → 256)**Deep Feed-Forward Network** - 深度前馈网络

  - 残差块2: [Linear(256 → 128) → BatchNorm → ReLU → Linear(128 → 128) → BatchNorm] + Skip(256 → 128)- **Architecture**: Sequential layers with batch normalization and dropout

  - 输出层: Linear(128 → 64) → BatchNorm → ReLU → Linear(64 → 32) → ReLU → Linear(32 → 1)- **Layers**: 

- **训练参数**:   - Linear(22 → 256) → BatchNorm → ReLU → Dropout(0.4)

  - 优化器: AdamW (lr=0.0005, weight_decay=1e-3)  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.35)

  - 损失函数: BCEWithLogitsLoss（类别加权）  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)

  - 训练轮数: 100 epochs, early_stopping patience=30  - Linear(64 → 32) → ReLU → Dropout(0.25)

  - 批次大小: 32  - Linear(32 → 1) → Sigmoid

- **Loss Function**: BCELoss (balanced dataset)

#### Australian信用数据集- **Optimization**: AdamW (lr=0.002, weight_decay=1e-3), ReduceLROnPlateau scheduler  

- **架构**: 轻量级神经网络 (Lightweight Network)- **Training**: 100 epochs (optimized), patience=20, batch_size=64

- **特点**: 简化结构，适应小样本数据- **Expected Accuracy**: 85%+

- **层结构**:- **Reference**: Deep Neural Networks for Credit Scoring - Khandani et al. (2010)

  - Linear(22 → 128) + BatchNorm + ReLU + Dropout(0.3)

  - Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.2)### UCI Credit Default Dataset (30,000 samples, 23 features)

  - Linear(64 → 32) + ReLU**Large-Scale Deep Network** - 大规模深度网络

  - Linear(32 → 1)- **Architecture**: Deep network optimized for large datasets

- **训练参数**: - **Layers**:

  - 优化器: AdamW (lr=0.001, weight_decay=1e-4)  - Linear(23 → 512) → BatchNorm → ReLU → Dropout(0.5)

  - 训练轮数: 100 epochs, early_stopping patience=25  - Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.45)

  - 批次大小: 32  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)

  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.35)

#### UCI信用数据集  - Linear(64 → 32) → ReLU → Dropout(0.3)

- **架构**: 深度全连接网络 (Deep Fully-Connected Network)  - Linear(32 → 1) → Sigmoid

- **特点**: 大容量模型，适应大规模数据- **Loss Function**: BCELoss with focal loss characteristics for large-scale training

- **层结构**:- **Optimization**: AdamW (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler

  - Linear(23 → 256) + BatchNorm + ReLU + Dropout(0.3)- **Training**: 100 epochs (optimized), patience=25, batch_size=128  

  - Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.2)- **Expected Accuracy**: 82%+

  - Linear(128 → 64) + BatchNorm + ReLU + Dropout(0.1)- **Reference**: Large-scale Credit Scoring - Lessmann et al. (2015)

  - Linear(64 → 32) + ReLU

  - Linear(32 → 1)---

- **训练参数**: 

  - 优化器: AdamW (lr=0.001, weight_decay=1e-4)## 📊 Four-Model Comparison Framework

  - 训练轮数: 150 epochs, early_stopping patience=30

  - 批次大小: 128本系统训练并对比以下四种模型：



### 2. SHAP特征重要性分析### 1. Teacher Model (教师模型)

- **架构**: 数据集特定的PyTorch深度神经网络

**SHAP (SHapley Additive exPlanations)** 是一种基于博弈论的模型解释方法：- **特点**: 高预测准确性，复杂度高

- **目的**: 作为知识蒸馏的源模型

- **核心思想**: 计算每个特征对预测结果的边际贡献

- **优势**: ### 2. Baseline Decision Tree (基准决策树)

  - 理论基础坚实（Shapley值的唯一性）- **架构**: 标准scikit-learn DecisionTreeClassifier

  - 模型无关（可应用于任何机器学习模型）- **特点**: 高可解释性，简单结构

  - 局部和全局解释兼顾- **目的**: 提供基准性能对比

- **实现流程**:

  1. 使用TreeExplainer分析教师模型### 3. All-Feature Distillation (全特征蒸馏)

  2. 计算每个特征的平均|SHAP值|- **架构**: 使用全部特征的知识蒸馏决策树

  3. 按重要性排序，选择Top-k特征- **特点**: 平衡准确性和可解释性

  4. 生成可视化报告- **目的**: 完整特征空间下的知识迁移



### 3. 知识蒸馏（Knowledge Distillation）### 4. Top-k Feature Distillation (Top-k特征蒸馏)

- **架构**: 基于SHAP Top-k特征的知识蒸馏决策树

**核心机制**: 将教师模型的"软知识"迁移到学生模型- **特点**: 精简特征集，高效解释

- **目的**: 最优特征子集下的知识迁移

- **软标签生成**:

  ```---

  soft_labels = softmax(teacher_logits / T)

  ```## 🔬 Knowledge Distillation Process

  其中T为温度参数，控制概率分布的平滑程度

### 核心技术参数

- **蒸馏损失函数**:- **Temperature Scaling**: T ∈ {1, 2, 3, 4, 5} for soft label generation

  ```- **Loss Combination**: α ∈ {0.0, 0.1, ..., 1.0} for balancing hard and soft losses

  L_distill = α * L_hard + (1-α) * L_soft- **Dynamic Feature Selection**: 

  ```  - German Dataset: k ∈ {5, 6, 7, ..., 54}

  - L_hard: 硬标签损失（真实标签）  - Australian Dataset: k ∈ {5, 6, 7, ..., 22}

  - L_soft: 软标签损失（教师模型输出）  - UCI Dataset: k ∈ {5, 6, 7, ..., 23}

  - α: 加权系数（0到1之间）- **Tree Optimization**: Optuna-based hyperparameter tuning for decision trees

- **Decision Tree Depth**: max_depth ∈ {3, 4, 5, ..., 10}

- **学生模型**: 决策树（DecisionTreeClassifier）

  - 参数优化: 使用固定参数确保可重复性### 蒸馏过程

  - 固定参数: max_depth(由实验确定), min_samples_split=2, min_samples_leaf=11. **Teacher Training**: 训练数据集特定的深度神经网络

  - 蒸馏实现: 使用软标签最大概率作为样本权重2. **SHAP Analysis**: 计算特征重要性并排序

3. **Knowledge Transfer**: 通过温度缩放软标签进行知识迁移

### 4. 消融实验（Ablation Study）4. **Student Optimization**: 基于混合损失函数优化决策树学生模型

5. **Rule Extraction**: 从训练好的决策树中提取可解释规则

系统性分析各参数对模型性能的影响：

---

- **Top-k特征数量**: k ∈ [5, n_features]  - Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)

- **温度参数**: T ∈ {1, 2, 3, 4, 5}  - Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.35)

- **蒸馏权重**: α ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}  - Linear(64 → 32) → ReLU → Dropout(0.3)

- **决策树深度**: max_depth ∈ {4, 5, 6, 7, 8}  - Linear(32 → 1) → Sigmoid

- **Loss Function**: BCELoss with focal loss characteristics for large-scale training

---- **Optimization**: AdamW (lr=0.001, weight_decay=1e-4), ReduceLROnPlateau scheduler

- **Training**: 300 epochs, patience=25, batch_size=128  

## 🚀 快速开始- **Expected Accuracy**: 82%+



### 环境要求## 📊 Four-Model Comparison Framework



```bash1. **Teacher Model**: Dataset-specific deep neural networks (architectures above)

Python >= 3.82. **Baseline Decision Tree**: Standard scikit-learn DecisionTreeClassifier  

PyTorch >= 1.10.03. **All-Feature Distillation**: Knowledge distillation using complete feature set

scikit-learn >= 1.0.04. **Top-k Feature Distillation**: SHAP-guided feature selection for targeted distillation

shap >= 0.40.0

pandas >= 1.3.0## 🔬 Knowledge Distillation Process

numpy >= 1.21.0

matplotlib >= 3.4.0- **Temperature Scaling**: T ∈ {1, 2, 3, 4, 5} for soft label generation

seaborn >= 0.11.0- **Loss Combination**: α ∈ {0.0, 0.1, ..., 1.0} for balancing hard and soft losses

openpyxl >= 3.0.0- **Feature Selection**: Dynamic k ranges (German: 5-54, Australian: 5-22, UCI: 5-23)

optuna >= 3.0.0- **Tree Optimization**: Optuna-based hyperparameter tuning for decision trees

tqdm >= 4.62.0

```Financial innovation/



### 安装依赖├── data/                          # Dataset storage- **Knowledge Distillation**: 将教师模型知识迁移到学生模型



```bash│   ├── german_credit.csv          # German Credit Dataset

pip install torch scikit-learn shap pandas numpy matplotlib seaborn openpyxl optuna tqdm xlrd

```│   ├── australian_credit.csv      # Australian Credit Dataset### 🎯 Advanced Knowledge Distillation- **PyTorch Neural Networks**: 高性能深度学习教师模型



### 运行实验│   └── uci_credit.xls            # UCI Taiwan Credit Dataset



```bash├── results/                       # Output files (generated)- **Temperature-scaled Soft Labels**: Configurable temperature parameter (T ∈ {1,2,3,4,5})- **Decision Tree**: 可解释性强的学生模型

# 运行完整实验流程

python main.py│   ├── model_comparison_*.xlsx    # Performance comparison table

```

│   ├── shap_feature_importance.png # SHAP visualization- **Hybrid Loss Function**: Balanced combination of hard and soft label losses (α ∈ {0.0,0.1,...,1.0})

实验流程包括：

1. 数据加载和预处理│   └── best_topk_rules_*.txt      # Extracted decision rules

2. 教师模型训练（神经网络）

3. SHAP特征重要性分析├── trained_models/               # Saved models (generated)- **Multi-depth Decision Trees**: Adaptive tree depth optimization (3-10 levels)---

4. 知识蒸馏和模型训练

5. 消融实验分析│   ├── teacher_model_*.pth       # PyTorch teacher models

6. 结果可视化和报告生成

│   ├── teacher_model_*.pkl       # Scikit-learn format

### 清理结果

│   └── teacher_model_*.json      # Model metadata

```bash

# 清空results文件夹├── main.py                       # Main execution pipeline### 📊 SHAP-Based Feature Selection  ## 系统架构

python clean_results.py

```├── data_preprocessing.py         # Data loading and preprocessing



---├── neural_models.py             # Neural network architectures- **Intelligent Feature Ranking**: TreeExplainer-based SHAP value computation



## 📊 实验结果├── distillation_module.py       # Knowledge distillation implementation



### 性能对比示例├── shap_analysis.py             # SHAP feature importance analysis- **Top-k Selection**: Systematic evaluation of k ∈ {5,6,7,8} most important features```



| 数据集 | 教师模型准确率 | 基线决策树 | Top-k决策树 | 特征压缩率 |├── result_manager.py            # Output management and reporting

|--------|---------------|-----------|------------|----------|

| German | 76.5% | 74.2% | 76.0% (k=35) | 35.2% |├── teacher_model_saver.py       # Model serialization utilities- **Cross-Dataset Analysis**: Comparative feature importance across datasets├── data/                          # 数据集

| Australian | 87.0% | 85.5% | 87.2% (k=12) | 45.5% |

| UCI | 82.3% | 80.1% | 81.9% (k=20) | 13.0% |└── README.md                    # This documentation



### 可视化输出```│   ├── uci_credit.xls            # UCI信用卡数据集



1. **SHAP特征重要性图**: 

   - `shap_german_features.png` - German数据集特征重要性

   - `shap_australian_features.png` - Australian数据集特征重要性## � SHAP Feature Analysis

   - `shap_uci_features.png` - UCI数据集特征重要性

### SHAP方法特点

2. **消融实验分析图**:- **TreeExplainer**: 针对决策树模型优化的SHAP解释器

   - `topk_ablation_visualization_*.png` - Top-k特征数量影响分析- **全数据集分析**: 使用训练+验证+测试的完整数据集

   - `depth_ablation_visualization_*.png` - 决策树深度影响分析- **精确特征排序**: 基于平均绝对SHAP值进行特征重要性排名

- **可视化输出**: 生成Top-20特征的对比图表

3. **决策规则文件**:

   - `best_topk_rules_*.txt` - 最优Top-k模型的可解释决策规则### 特征重要性可视化

   - `best_all_feature_rules_*.txt` - 全特征模型的决策规则- **数据集顺序**: German → Australian → UCI

- **颜色方案**: 浅蓝色系 → 浅绿色系 → 浅橙色系

4. **性能对比表**:- **特征数量**: 每个数据集显示Top-20重要特征

   - `model_comparison_*.xlsx` - 各模型详细性能指标对比- **真实特征名**: 使用英文原始特征名而非编码名



------



## 🔍 代码模块说明## 🔧 Core Modules



### main.py### 1. Data Preprocessing (`data_preprocessing.py`)

主程序入口，协调整个实验流程：- **功能**: 加载和预处理三个信用数据集

- 数据加载和划分- **核心特性**:

- 教师模型训练  - 标准化的数据加载和train/validation/test划分

- SHAP分析  - 分类变量的特征编码

- 知识蒸馏实验  - 数据缩放和标准化

- 结果汇总和可视化  - 特征名追踪以保证可解释性



### data_preprocessing.py### 2. Neural Network Models (`neural_models.py`)

数据预处理模块：- **功能**: 定义和训练教师神经网络

- 加载三个信用数据集- **架构特点**:

- 数据清洗和特征工程  - 带残差连接的高级前馈网络

- 标准化处理  - 批量标准化和dropout正则化

- 训练/验证/测试集划分  - 自适应学习率调度

  - 早停和模型检查点

### neural_models.py

神经网络教师模型：### 3. SHAP Analysis (`shap_analysis.py`)

- CreditNet类（残差网络架构）- **功能**: 使用SHAP进行特征重要性分析

- 三个数据集的专用模型配置- **处理流程**:

- 训练和评估功能  - 为每个数据集训练优化的决策树

- Early stopping和学习率调度  - 使用TreeExplainer计算SHAP值

  - 生成top-k特征排名

### shap_analysis.py  - 创建带有正确特征名的可视化

SHAP特征重要性分析：

- TreeExplainer集成### 4. Knowledge Distillation (`distillation_module.py`)

- 特征重要性计算和排序- **功能**: 从教师模型向学生模型转移知识

- Top-k特征选择- **实现细节**:

- 可视化生成（单独保存每个数据集）  - 温度缩放的软标签生成

  - 混合损失函数(硬标签+软标签)

### distillation_module.py  - 基于SHAP的top-k特征选择

知识蒸馏核心模块：  - 从训练好的树中提取决策规则

- 软标签提取

- 温度缩放### 5. Result Management (`result_manager.py`)

- 决策树蒸馏训练- **功能**: 组织和导出结果

- Top-k和全特征实验- **输出内容**:

- 多进程并行优化  - 基于Excel的性能对比

  - 决策规则文本文件

### ablation_analyzer.py  - 模型性能指标

消融实验分析：

- 实验结果记录### 6. Ablation Analysis (`ablation_analyzer.py`)

- Top-k和深度消融可视化- **功能**: 消融实验分析和可视化

- 最优配置识别- **输出图表**:

- 报告生成  - Top-k特征数量消融实验

  - 决策树深度消融实验

### result_manager.py  - 1×2布局的简化图表

结果管理模块：

- 决策规则提取---

- Excel报告生成

- 性能对比表## 📈 Datasets

- 结果文件组织

系统在三个广泛使用的信用评分基准数据集上进行评估：

### clean_results.py

结果清理工具：### 1. German Credit Dataset (1,000 samples, 54 features)

- 安全删除results文件夹内容- **来源**: UCI Machine Learning Repository

- 文件占用检测- **任务**: 二分类(好/坏信用风险)

- 批量清理功能- **特征**: 人口统计学、账户状态、信用历史



---### 2. Australian Credit Approval Dataset (690 samples, 22 features)

- **来源**: UCI Machine Learning Repository  

## 🎓 理论背景- **任务**: 二分类(批准/拒绝信用)

- **特征**: 匿名化的申请人属性

### 知识蒸馏（Knowledge Distillation）

- **提出者**: Hinton et al., 2015### 3. Taiwan Credit Card Default Dataset (30,000 samples, 23 features)

- **核心思想**: 复杂模型（教师）→简单模型（学生）知识迁移- **来源**: UCI Machine Learning Repository

- **关键技术**: 软标签、温度缩放、损失函数设计- **任务**: 二分类(违约/非违约)

- **特征**: 支付历史、账单金额、人口统计数据

### SHAP值（Shapley Additive Explanations）

- **理论基础**: 博弈论中的Shapley值---

- **特性**: 唯一性、一致性、局部准确性

- **应用**: 模型解释、特征选择、异常检测## 🚀 Installation & Usage



### 决策树（Decision Tree）### Prerequisites

- **优势**: 高可解释性、非线性建模、无需特征缩放```bash

- **挑战**: 易过拟合、不稳定性pip install torch scikit-learn pandas numpy matplotlib seaborn shap openpyxl optuna tqdm

- **本项目优化**: 知识蒸馏、固定参数、样本加权```



---### Quick Start

```bash

## 📈 性能优化技巧# Clone the repository

git clone https://github.com/lidengjia1/shapGuided_KnowledgeDistilling.git

### 1. 随机种子控制cd shapGuided_KnowledgeDistilling

所有随机过程统一设置seed=42，确保实验100%可重复：

- NumPy随机数生成器# Run the complete pipeline

- PyTorch随机数生成器python main.py

- CUDA随机数生成器```

- Python内置random模块

- 环境变量PYTHONHASHSEED### Expected Outputs

- cuDNN确定性算法运行完成后，将生成以下核心文件：



### 2. 并行计算1. **模型性能对比表** (`results/model_comparison_*.xlsx`)

- 使用多进程/多线程加速消融实验   - 四种模型的性能指标

- 自动检测CPU核心数并合理分配   - 准确率、F1分数、精确率、召回率

- Windows平台特殊优化（spawn模式）   - 每种配置的最佳超参数



### 3. 内存优化2. **SHAP特征重要性图** (`results/shap_feature_importance.png`)

- 批次处理大数据集   - 三个数据集的可视化对比

- 及时释放中间变量   - 每个数据集的Top-20重要特征

- 非交互式matplotlib后端   - 英文标签和正确的特征名



### 4. 数值稳定性3. **Top-k决策规则** (`results/best_topk_rules_*.txt`)

- 使用BCEWithLogitsLoss避免数值溢出   - 从最佳模型提取的决策树规则

- 梯度裁剪防止梯度爆炸   - 特征重要性排名

- BatchNormalization稳定训练   - 模型性能详情



---4. **消融实验图** (`results/*_ablation_study_analysis_*.png`)

   - Top-k特征数量消融实验

## 🐛 常见问题   - 决策树深度消融实验



### Q1: 为什么每次运行结果会有差异？---

**A**: 现已完全修复。通过设置完整的随机种子（NumPy、PyTorch、CUDA、random模块、PYTHONHASHSEED、cuDNN），确保实验完全可重复（差异=0）。

## � Experimental Configuration

### Q2: 如何调整Top-k特征数量范围？

**A**: 在`main.py`中修改k_ranges参数：### 参数空间

```python- **Top-k特征数**: 

k_ranges = {  - German Dataset: k ∈ {5, 6, ..., 54}

    'german': (5, 54),      # 从5到54  - Australian Dataset: k ∈ {5, 6, ..., 22}

    'australian': (5, 22),  # 从5到22  - UCI Dataset: k ∈ {5, 6, ..., 23}

    'uci': (5, 23)          # 从5到23- **蒸馏温度**: T ∈ {1, 2, 3, 4, 5}

}- **损失权重**: α ∈ {0.0, 0.1, 0.2, ..., 1.0}

```- **树深度**: max_depth ∈ {3, 4, 5, 6, 7, 8, 9, 10}



### Q3: 如何修改消融实验参数？### 评估指标

**A**: 在`main.py`中调整以下参数：- **准确率 (Accuracy)**: 正确预测的比例

```python- **F1分数 (F1-Score)**: 精确率和召回率的调和平均

temperature_range = [1, 2, 3, 4, 5]  # 温度参数- **精确率 (Precision)**: 正预测中的正确比例

alpha_range = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 蒸馏权重- **召回率 (Recall)**: 实际正例中的预测正确比例

max_depth_range = [4, 5, 6, 7, 8]  # 决策树深度

```### 并发优化

- **Windows平台**: 使用min(4, cpu_count//2)个并发进程

### Q4: 如何加快实验速度？- **Linux/Mac平台**: 使用min(cpu_count-1, cpu_count)个并发进程

**A**: - **进度显示**: 集成tqdm进度条，实时显示训练进度

1. 减少消融实验的参数组合

2. 使用GPU加速神经网络训练---

3. 增加并行进程数（如果CPU核心充足）

4. 减少数据集规模（用于快速测试）



### Q5: 内存不足怎么办？- **v2.0**: Complete refactoring with SHAP-guided distillation# Run complete analysis pipeline- **Accuracy**: 分类准确率

**A**: 

1. 减小批次大小（batch_size）- **v1.9**: Enhanced neural network architectures

2. 减少神经网络层数或隐藏单元数

3. 分批处理消融实验- **v1.8**: Improved feature name handling and visualizationpython main.py- **Precision**: 精确率

4. 关闭不必要的可视化

- **v1.7**: Added comprehensive result management

### Q6: 消融图标注重叠怎么办？

**A**: 已修复。系统现在根据k值和depth值智能调整标注位置，避免重叠。- **v1.6**: Optimized knowledge distillation pipeline```- **Recall**: 召回率



---



## 📝 实验可重复性---- **F1-Score**: F1分数



本项目遵循严格的可重复性标准：



✅ **完整的随机种子控制**: 所有随机过程均设置固定种子  *This project represents cutting-edge research in explainable AI for financial applications, combining the power of deep learning with the interpretability requirements of financial decision-making.*This will generate three key outputs:- **AUC**: ROC曲线下面积

✅ **详细的参数记录**: 所有实验参数自动保存  

✅ **版本化的依赖**: 明确指定库版本要求  

✅ **标准化的数据处理**: 数据预处理流程固定  

✅ **自动化的实验流程**: 一键运行完整实验  1. **Model Comparison Table** (`results/model_comparison_*.xlsx`)---

✅ **防重复生成机制**: 避免相同文件重复生成  

   - Performance metrics for all four model types

---

   - Statistical significance tests## 环境配置

## 🔄 更新日志

   - Hyperparameter configurations

### v2.0.0 (2025-01-09)

- ✅ 修复随机种子控制，实现100%可重复性### 依赖安装

- ✅ 优化消融图标注位置，智能避免重叠

- ✅ 删除冗余的图像生成代码2. **SHAP Feature Importance Visualization** (`results/shap_feature_importance.png`)```bash

- ✅ 重构README为中文版本

- ✅ 添加文件存在性检查，防止重复生成   - Top-8 features for each datasetpip install torch pandas scikit-learn xgboost shap matplotlib openpyxl numpy

- ✅ 改进SHAP可视化配色方案

   - Comparative importance scores```

### v1.0.0

- 初始版本发布   - Cross-dataset feature analysis

- 实现基础知识蒸馏功能

- SHAP特征选择### 运行系统

- 消融实验分析

3. **Top-k Decision Rules** (`results/best_topk_rules_*.txt`)```bash

---

   - Interpretable IF-THEN rules from best distilled modelspython main.py

## 🤝 贡献指南

## 📈 Key Findings

欢迎贡献代码、报告问题或提出改进建议！

### 主要实验结果

### 贡献方式- **Top-k特征蒸馏**达到与全特征模型相当的准确率

1. Fork本仓库- **SHAP引导的特征选择**显著提升了模型可解释性  

2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)- **知识蒸馏**有效缩小了准确率与可解释性之间的差距

3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)- **温度缩放和损失加权**是有效蒸馏的关键技术

4. 推送到分支 (`git push origin feature/AmazingFeature`)

5. 开启Pull Request### 性能基准测试



---| Dataset | Teacher (DNN) | Baseline Tree | All-Feature Distill | Top-k Distill |

|---------|---------------|---------------|-------------------|---------------|

## 📄 许可证| German | 0.75-0.78 | 0.70-0.73 | 0.73-0.76 | 0.74-0.77 |

| Australian | 0.85-0.88 | 0.82-0.85 | 0.84-0.87 | 0.85-0.88 |

本项目采用MIT许可证。| UCI Taiwan | 0.80-0.83 | 0.76-0.79 | 0.78-0.81 | 0.79-0.82 |



---*注：范围反映不同超参数配置下的性能变化*



## 📧 联系方式---



如有问题或建议，请通过GitHub Issues联系。## 📚 Technical References



---本项目基于以下前沿研究成果：



## 🙏 致谢### 知识蒸馏相关

- **Neural Network Distillation**: Hinton et al. (2015) - 温度缩放和软标签训练

感谢以下开源项目和研究工作：- **Tabular Data Distillation**: 针对表格数据的知识蒸馏优化

- PyTorch团队

- SHAP库作者 (Scott Lundberg)### SHAP可解释AI

- scikit-learn社区- **SHAP Values**: Lundberg & Lee (2017) - TreeExplainer精确特征重要性计算

- Knowledge Distillation相关研究者- **Feature Selection**: 基于SHAP的智能特征选择策略



---### 神经架构设计

- **Residual Networks**: He et al. (2016) - 残差连接改善梯度流

## 📚 参考文献- **Credit Scoring DNNs**: 针对信用评分优化的神经网络架构



1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.### 金融机器学习

- **Financial ML**: Lopez de Prado (2018) - 金融风险评估和可解释建模

2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.- **Regulatory Compliance**: 符合金融监管要求的可解释AI方法



3. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.---



4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.## 🔄 Version History



---### v2.0.0 (Current) - Enhanced Performance

- ✅ 优化教师模型架构，提升German数据集准确率至75%+

**最后更新**: 2025年1月9日  - ✅ 减少训练epochs，提高训练效率

**版本**: 2.0.0  - ✅ 简化消融实验图表为1×2布局

**状态**: ✅ 生产就绪- ✅ 改进SHAP可视化配色方案

- ✅ 禁用文件自动清理功能
- ✅ 增强Windows平台并发支持

### v1.0.0 - Initial Release
- ✅ 基础知识蒸馏框架
- ✅ SHAP特征重要性分析
- ✅ 三数据集支持
- ✅ 基础可视化功能

---

## 📧 Contact Information

**Primary Author**: Li Dengjia  
**Email**: lidengjia@hnu.edu.cn  
**Institution**: Hunan University  
**Research Focus**: Financial AI, Knowledge Distillation, Explainable Machine Learning

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- UCI Machine Learning Repository for providing the benchmark datasets
- SHAP library developers for interpretability tools
- PyTorch team for the deep learning framework  
- Research community for advances in knowledge distillation and explainable AI

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@misc{li2025shap_distillation,
  title={SHAP-Guided Knowledge Distillation for Credit Scoring},
  author={Li, Dengjia and [Co-authors]},
  year={2025},
  institution={Hunan University},
  note={A comprehensive framework for interpretable credit scoring using SHAP-guided knowledge distillation}
}
```

---

*This project represents ongoing research in interpretable machine learning for financial applications. Contributions and collaborations are welcome.*

**Last Updated**: September 16, 2025  
**Version**: v2.0.0
