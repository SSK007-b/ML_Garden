# 🌱 ML Garden

**ML Garden** is a curated collection of classical machine learning projects built on a **shared, reusable pipeline** for training, evaluation, and inference on tabular data.

Rather than experimenting with every possible algorithm, this repository emphasizes **clarity, reusability, and real-world ML system design**.

---

## 🎯 Repository Goals

- Demonstrate **end-to-end ML workflows** for tabular problems  
- Show how **multiple models** can be trained and evaluated using a **common codebase**  
- Enforce a clean separation of concerns:
  - Data preprocessing  
  - Feature engineering  
  - Model definition  
  - Training & evaluation  
  - Inference  
- Promote **reproducible, well-documented, production-aware ML projects**

---

## 📦 What’s Inside

This repository contains multiple **standalone ML projects**, all powered by shared utilities to ensure consistency and comparability.

### Included Models

- **Voting Ensemble** – Hybrid model combining multiple learners  
- **Random Forest** – Robust tree-based ensemble  
- **XGBoost** – High-performance gradient boosting for tabular data  

All models follow **identical preprocessing and evaluation standards**, enabling fair model comparison.

---

## 📁 Repository Structure

```bash
ML_garden/
│
├── common/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   └── utils.py
│
├── projects/
│   ├── loan_approval/
│   ├── pistachio_prediction/
│   └── heart_disease_prediction/
│
├── requirements.txt
├── pyproject.toml
└── README.md
```
- common/ contains reusable pipeline components
- projects/ hosts independent ML use cases built on top of the shared pipeline

## 🧠 Design Philosophy
*Why not include every ML algorithm?*
This repository intentionally focuses on a small, representative set of models that capture different modeling philosophies:
- Interpretable models for transparency
- Tree-based ensembles for non-linear relationships
- Gradient boosting for strong tabular performance
Algorithms with overlapping characteristics are deliberately excluded to keep the codebase focused, maintainable, and meaningful.

## 🔁 Shared ML Pipeline
All projects follow the same high-level workflow:
- Data loading & validation
- Column normalization & preprocessing
- Feature engineering
- Model training
- Evaluation using multiple metrics
- Model persistence
- Inference on new data
This standardized pipeline lives in the common/ directory and is reused across all projects.

## 📊 Experiments & Model Comparison
Each project includes an exploratory notebooks folder:
```bash
projects/<project-name>/notebooks/
```
These notebooks are used for:
- Comparing multiple models on the same dataset
- Understanding performance trade-offs
- Supporting model selection decisions
Notebooks are exploratory only and are not part of the production pipeline.

## 🗂️ Datasets
- Full training datasets are not included
- Each project README provides dataset sources and instructions
- Small, anonymized sample datasets are included for testing

This approach aligns with best practices for data licensing, reproducibility, and repository cleanliness

## ▶️ How to Run a Project
From the repository root:
```bash
pip install -r requirements.txt
pip install -e .
```
Navigate to a specific project:
```bash
cd projects/loan_approval
python -m projects.loan_approval.train
```
Refer to each project’s README for project-specific configuration and details.

## 🚀 Future Extensions
Planned enhancements include:
- Configuration-driven model selection
- Unified CLI interface
- API-based inference (FastAPI / Django)
- Experiment tracking and versioning
- Automated testing for ML pipelines

## 👤 Author
Built and maintained as part of a personal ML engineering portfolio, with emphasis on:
- clean, modular code
- reproducible workflows
- practical ML system design
- industry-aligned project structure

## 📌 Final Note

ML Garden is not a tutorial dump or a model zoo.

It is a carefully structured collection of applied ML projects, designed to reflect how real-world machine learning systems are built, evaluated, and maintained.