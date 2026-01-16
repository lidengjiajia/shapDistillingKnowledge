# SHAP-guided Adaptive Knowledge Distillation for Credit Scoring
# SHAP引导的自适应知识蒸馏信用评分系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for interpretable credit scoring using SHAP-guided knowledge distillation with theoretical foundations.

## 🎯 Key Features

- **Academic Baseline Models**: LR-Ridge, LR-Lasso, LR-ElasticNet, SVM-RBF, RF, GBDT, XGBoost, LightGBM, CatBoost
- **Neural Teacher Models**: MLP, ResNet, Transformer architectures
- **SAKD Framework**: SHAP-guided Adaptive Knowledge Distillation with theoretical proofs
- **SHAP Interpretability**: Feature importance, stability analysis, and visualizations
- **GPU Acceleration**: CUDA support for all deep learning and tree-based models
- **Systematic Ablation**: Temperature, alpha, and architecture ablation experiments

## 📊 Datasets

| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| German Credit | 1,000 | 20 | UCI |
| Australian Credit | 690 | 14 | UCI |
| Xinwang Credit | 17,884 | 100 | Chinese P2P |
| UCI Credit Card | 30,000 | 23 | UCI |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/credit-scoring-kd.git
cd credit-scoring-kd

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
scikit-learn>=1.0.0
xgboost>=1.7.0
lightgbm>=3.3.0
catboost>=1.0.0
shap>=0.41.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

### Run Experiments

```bash
# Run full experiment pipeline
python run_experiments.py --dataset german --gpu

# Available datasets: german, australian, xinwang, uci
```

## 📁 Project Structure

```
credit-scoring-kd/
├── src/                          # Source code
│   ├── data/                     # Data preprocessing
│   │   ├── __init__.py
│   │   ├── preprocessor.py       # DataPreprocessor class
│   │   └── dataset.py            # PyTorch Dataset
│   ├── models/                   # Model implementations
│   │   ├── __init__.py
│   │   ├── baselines.py          # Traditional ML baselines
│   │   ├── neural.py             # Neural network models
│   │   └── sota_baselines.py     # SOTA models (TabNet, etc.)
│   ├── distillation/             # Knowledge distillation
│   │   ├── __init__.py
│   │   ├── sakd_framework.py     # SAKD with theoretical proofs
│   │   └── advanced_distillation.py
│   ├── interpretability/         # SHAP analysis
│   │   ├── __init__.py
│   │   └── shap_analyzer.py      # SHAPAnalyzer class
│   └── utils/                    # Utilities
│       ├── config_manager.py
│       └── experiment_tracker.py
├── config/                       # Configuration files
│   └── experiment_config.yaml
├── data/                         # Datasets
│   ├── german_credit.csv
│   ├── australian_credit.csv
│   └── xinwang.csv
├── results/                      # Experiment outputs
├── visualization/                # Plotting utilities
│   └── ablation_plots.py
├── run_experiments.py            # Main experiment runner
└── README.md                     # This file
```

## 📐 Theoretical Foundations

### Theorem 1: Temperature-Interpretability Tradeoff

$$\mathbb{E}[\|p_S - p_T\|_2] \leq \frac{C_1}{\sqrt{\tau}} + C_2 \cdot \exp\left(-\frac{\tau}{\tau_0}\right)$$

### Theorem 2: Generalization Bound for SHAP-guided Distillation

$$\epsilon_S \leq \epsilon_T + O\left(\sqrt{\frac{k \cdot \log k}{n}}\right) + O\left(d_{\max}^{-1}\right) + O\left(\frac{1}{\tau}\right)$$

### Theorem 3: Feature Selection Consistency

$$P\left(|S_k \cap S_k^*| \geq (1-\delta)k\right) \geq 1 - 2\exp\left(-\frac{n\delta^2}{2}\right)$$

## 🔬 Baseline Models

| Model | Category | Reference |
|-------|----------|-----------|
| LR-Ridge | Linear | Hosmer & Lemeshow (2000) |
| LR-Lasso | Linear | Tibshirani (1996) |
| LR-ElasticNet | Linear | Zou & Hastie (2005) |
| SVM-RBF | Kernel | Cortes & Vapnik (1995) |
| RF | Ensemble | Breiman (2001) |
| GBDT | Ensemble | Friedman (2001) |
| XGBoost | Ensemble | Chen & Guestrin (2016) |
| LightGBM | Ensemble | Ke et al. (2017) |
| CatBoost | Ensemble | Prokhorenkova et al. (2018) |

## 📈 Ablation Experiments

| Dimension | Values | Purpose |
|-----------|--------|---------|
| Temperature (τ) | {1, 2, 4, 8, 16} | Theorem 1 validation |
| Alpha (α) | {0.3, 0.5, 0.7, 0.9} | Soft/hard target balance |
| Architecture | Tiny/Small/Medium/Large | Model complexity analysis |

## 🖥️ GPU Configuration

The framework automatically detects and uses GPU when available:

```python
# Automatic GPU detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# XGBoost GPU
xgb.XGBClassifier(tree_method='hist', device='cuda')

# LightGBM GPU
lgb.LGBMClassifier(device='gpu')

# CatBoost GPU
CatBoostClassifier(task_type='GPU')
```

## 📊 Example Results

### German Credit Dataset

| Model | AUC | Accuracy | F1 |
|-------|-----|----------|-----|
| LR-Ridge | 0.756 | 0.725 | 0.712 |
| XGBoost | 0.867 | 0.834 | 0.821 |
| CatBoost | 0.873 | 0.841 | 0.828 |
| **SAKD-Student** | **0.879** | **0.848** | **0.835** |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{author2024sakd,
  title={SHAP-guided Adaptive Knowledge Distillation for Interpretable Credit Scoring},
  author={Author, A. and Author, B.},
  journal={Financial Innovation},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

