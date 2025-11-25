# SHAP引导的知识蒸馏信用评分系统

基于SHAP特征重要性分析和知识蒸馏的信用评分模型优化框架。

## 特性

- **多模型对比**: 6种基线模型（Logistic、Logistic+L1、XGBoost、LightGBM、SVM、RandomForest）+ 深度神经网络
- **自动调优**: 使用Optuna进行超参数优化
- **知识蒸馏**: 将神经网络知识迁移到决策树，提升可解释性
- **SHAP分析**: 特征重要性可视化与Top-k特征选择
- **消融实验**: 系统化分析温度参数、蒸馏权重、决策树深度的影响

## 数据集

支持4个信用评分数据集：
- **German Credit** (1,000样本, 54特征)
- **Australian Credit** (690样本, 22特征)
- **UCI Taiwan Credit** (30,000样本, 23特征)
- **Xinwang Credit** (17,884样本, 100特征)

## 快速开始

### 安装依赖
```bash
pip install torch scikit-learn xgboost lightgbm optuna shap pandas numpy matplotlib seaborn tqdm openpyxl
```

### 运行实验
```bash
python main.py
```

## 项目结构

```
├── data/                        # 数据集目录
├── results/                     # 实验结果输出
├── main.py                      # 主程序入口
├── data_preprocessing.py        # 数据预处理
├── baseline_models.py           # 基线模型 (Logistic, XGB, LGBM, SVM, RF)
├── neural_models.py             # 神经网络教师模型
├── distillation_module.py       # 知识蒸馏核心
├── shap_analysis.py             # SHAP特征分析
├── ablation_analyzer.py         # 消融实验分析
└── result_manager.py            # 结果管理
```

## 核心流程

1. **数据预处理** → 标准化、划分训练/验证/测试集
2. **基线模型训练** → Optuna自动调优6种传统机器学习模型
3. **神经网络训练** → 深度残差网络作为教师模型
4. **SHAP分析** → 特征重要性排序
5. **知识蒸馏** → 迁移知识到决策树（学生模型）
6. **消融实验** → 分析超参数影响
7. **结果汇总** → 生成Excel报告和可视化图表

## 输出结果

实验结果保存在 `results/` 目录：
- **模型性能对比表** (Excel) - 所有模型的准确率、F1分数、精确率、召回率
- **SHAP特征重要性图** - 各数据集的Top特征可视化
- **消融实验图表** - Top-k和深度参数影响分析
- **决策规则文件** - 可解释的IF-THEN规则提取

## 评估指标

所有模型评估指标保留4位小数：
- **Accuracy** (准确率)
- **Precision** (精确率)
- **Recall** (召回率)
- **F1-Score** (F1分数)

## 技术亮点

### 基线模型（使用Optuna调优）
- **Logistic Regression** - L2正则化逻辑回归
- **Logistic + L1** - L1正则化逻辑回归（稀疏特征选择）
- **XGBoost** - 梯度提升树
- **LightGBM** - 轻量级梯度提升
- **SVM** - 支持向量机（RBF核）
- **Random Forest** - 随机森林

### 神经网络架构
- **German**: 残差网络（2层残差块）+ BatchNorm + Dropout
- **Australian**: 深度前馈网络 + 正则化
- **UCI**: 大规模深度网络（6层）
- **Xinwang**: 深度残差网络（3层残差块，953K参数）

### 知识蒸馏技术
- **温度缩放**: T ∈ {1, 2, 3, 4, 5}
- **损失函数**: α·L_hard + (1-α)·L_soft
- **特征选择**: 基于SHAP的Top-k特征动态选择
- **决策树优化**: max_depth ∈ {4, 5, 6, 7, 8}

## 参考文献

- Hinton et al. (2015) - Distilling the Knowledge in a Neural Network
- Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions
- arXiv:2411.17783 - Kolmogorov-Arnold Networks for Credit Scoring
- arXiv:2412.02097 - Hybrid KAN and gMLP Models for Financial Data

## 许可证

MIT License

