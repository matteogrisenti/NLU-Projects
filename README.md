# 🚀 Natural Language Understanding & Language Modeling Projects

A comprehensive portfolio showcasing advanced implementations of **Language Modeling (LM)** and **Natural Language Understanding (NLU)** tasks using state-of-the-art deep learning techniques.

## 📋 Project Overview

This repository demonstrates practical expertise in building neural network architectures for language processing tasks. The project is divided into two main components:

- **Language Modeling (LM)**: Predictive text models trained on Penn Treebank
- **Natural Language Understanding (NLU)**: Intent and slot classification systems

Both sections include multiple experimental approaches, hyperparameter tuning, and comprehensive performance evaluations.

---

## 🎯 Key Features

✨ **Multiple Model Architectures**
- RNN, LSTM, and LSTM with Dropout implementations
- Various optimization techniques (SGD, Adam, AdamW)
- Advanced strategies: Weight Tying, Variational Dropout, Neural Tangent Alignment

📊 **Comprehensive Experimentation**
- Systematic hyperparameter exploration
- Performance metrics tracking (perplexity, accuracy, F1-score)
- Visualization and analysis tools

🔧 **Production-Ready Code**
- Well-documented and modular design
- Proper error handling and logging
- Easy-to-use training and evaluation pipelines

---

## 📁 Project Structure

```
NL-Project/
├── LM/                          # Language Modeling Tasks
│   ├── A/                       # Basic Models (RNN, LSTM, LSTM-Dropout)
│   │   ├── model.py            # Model architectures
│   │   ├── main.py             # Training & evaluation pipeline
│   │   ├── functions.py        # Training utilities
│   │   ├── utils.py            # Data loading & preprocessing
│   │   ├── plots.py            # Results visualization
│   │   ├── bin/                # Pre-trained model weights
│   │   ├── dataset/            # Penn Treebank corpus
│   │   ├── results/            # Evaluation metrics (dev, test)
│   │   └── plots/              # Generated performance plots
│   │
│   └── B/                       # Advanced Models (NTAvSGD, VarDropout, WeightTying)
│       ├── model.py            # Advanced architectures
│       ├── main.py             # Training pipeline
│       ├── functions.py        # Training utilities
│       ├── utils.py            # Data utilities
│       ├── plot.py             # Visualization
│       ├── bin/                # Pre-trained weights
│       └── results/            # Performance metrics
│
├── NLU/                         # Natural Language Understanding Tasks
│   ├── A/                       # Intent & Slot Tagging (Basic)
│   │   ├── model.py            # BiLSTM model
│   │   ├── main.py             # Training pipeline
│   │   ├── conll.py            # CoNLL format handling
│   │   ├── functions.py        # Utilities
│   │   ├── plot.py             # Results visualization
│   │   └── dataset/            # Training data
│   │
│   └── B/                       # NLU with Advanced Techniques
│       ├── model.py            # Enhanced models
│       ├── main.py             # Advanced training
│       ├── conll.py            # CoNLL format
│       ├── functions.py        # Utilities
│       └── dataset/            # Corpus
│
└── requirements.txt            # Python dependencies
```

---

## 🧠 Models Implemented

### Language Modeling (LM)

**Part A - Foundational Models:**
| Model | Architecture | Optimizer | Key Features |
|-------|--------------|-----------|--------------|
| RNN | 1-layer RNN | SGD | Baseline recurrent network |
| LSTM | 1-layer LSTM | SGD | Handles long-term dependencies |
| LSTM-Dropout | LSTM + Regularization | SGD | Dropout for better generalization |
| ADAM | Multi-layer RNN | AdamW | Adaptive learning rates |

**Part B - Advanced Models:**
| Model | Technique | Purpose |
|-------|-----------|---------|
| Weight Tying | Shared embeddings | Parameter efficiency |
| Variational Dropout | Consistent dropout masks | Improved regularization |
| NTAvSGD | Neural Tangent Alignment | Better convergence |

### Natural Language Understanding (NLU)

**Intent & Slot Tagging System:**
- Intent Classification: Multi-class classification
- Slot Tagging: Sequence labeling with BiLSTM
- Models support both basic and advanced variants

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Quick Start

1. **Clone and navigate to the project:**
```bash
cd NL-Project
```

2. **Create a virtual environment:**
```bash
python -m venv myenv
source myenv/Scripts/activate  # On Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch: Deep learning framework
- NumPy & Pandas: Data manipulation
- Matplotlib & Seaborn: Visualization
- scikit-learn: ML utilities
- Transformers: Pre-trained models support

---

## 🚀 Usage

### Training Language Models

```bash
cd LM/A
python main.py  # Trains RNN/LSTM models
```

Uncomment specific training blocks in `main.py` to train different models or modify hyperparameters.

### Training NLU Models

```bash
cd NLU/A
python main.py  # Trains Intent & Slot classification
```

### Evaluating Models

Pre-trained models are available in `bin/` directories:
```bash
# Models are automatically loaded for evaluation
python main.py  # Runs evaluation on test set
```

Results are saved to `results/` with metrics in CSV format.

---

## 📊 Results & Evaluation

### Language Modeling Metrics
- **Perplexity**: Measures prediction quality on unseen text
- **Cross-Entropy Loss**: Training efficiency metric
- Results stored in: `LM/A/results/` and `LM/B/results/`

### NLU Metrics
- **Intent Accuracy**: Percentage of correctly classified intents
- **Slot F1-Score**: Weighted harmonic mean of precision/recall
- **Classification Reports**: Detailed per-class performance
- Results stored in: `NLU/A/results/` and `NLU/B/results/`

### Visualizations
Automatic plot generation for:
- Training/validation curves
- Loss progression
- Model comparisons
- Hyperparameter sensitivity analysis

---

## 🎓 Key Learnings & Technical Highlights

✅ **Deep Learning Fundamentals**
- Sequence modeling with RNNs, LSTMs
- Gradient flow and backpropagation through time (BPTT)
- Regularization techniques (Dropout, L2)

✅ **Advanced Techniques**
- Embedding layers and weight sharing
- Attention mechanisms for NLU tasks
- Hyperparameter optimization strategies

✅ **Software Engineering Practices**
- Modular, reusable code architecture
- Configuration management via dictionaries
- Proper data pipeline design
- Reproducible experiments with random seeds

✅ **Data Processing**
- Tokenization and vocabulary management
- Batch processing with DataLoaders
- Train/validation/test split methodology
- Handling variable-length sequences with padding

---

## 📈 Performance Highlights

This project demonstrates:
- **Model Selection**: Choosing appropriate architectures for different tasks
- **Hyperparameter Tuning**: Systematic exploration of learning rates, hidden sizes, batch sizes
- **Regularization**: Dropout, gradient clipping, early stopping
- **Optimization**: Comparing SGD, Adam, and AdamW optimizers
- **Evaluation**: Rigorous testing on held-out datasets

---

## 💡 Future Enhancements

Potential extensions for this project:
- [ ] Transformer-based models (BERT, GPT)
- [ ] Multi-task learning for joint intent+slot prediction
- [ ] Domain adaptation techniques
- [ ] Ensemble methods for improved performance
- [ ] Interactive API for model inference

---

## 📚 References & Technologies

**Frameworks & Libraries:**
- PyTorch for neural network implementation
- Penn Treebank dataset for language modeling
- Standard NLU benchmark datasets

**Techniques Applied:**
- Sequence-to-sequence learning
- Attention mechanisms
- Dropout regularization
- Gradient clipping

---

## 👤 Author

**Matteo Grisenti**  
GitHub: [@matteogrisenti02](https://github.com/matteogrisenti02)

---

## 📝 License

This project is part of coursework and is available for educational purposes.

---

## 🤝 How This Demonstrates Skills

This portfolio showcases:
- 🎯 **Problem Solving**: Implementing complex ML architectures from scratch
- 📊 **Data Science**: Preprocessing, training, evaluation workflows
- 💻 **Software Engineering**: Clean code, modularity, reproducibility
- 🧪 **Experimentation**: Systematic hyperparameter exploration
- 📈 **Communication**: Clear code organization and documentation

Perfect for demonstrating competency in machine learning roles involving NLP/NLU! 🚀
