# Product Review Sentiment Analysis

End-to-end machine learning pipeline for classifying e-commerce product review sentiment (positive, negative, neutral). Demonstrates the full ML workflow from raw text to a deployable scikit-learn model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## Overview

This project classifies customer review sentiment to support product quality monitoring, support triage, and marketplace analytics. Text features from review titles and bodies are combined with review length in a unified preprocessing pipeline.

| Task | Business question | Method |
|------|-------------------|--------|
| **1** | What is the sentiment distribution in reviews? | EDA & class balance |
| **2** | Which model performs best on text? | Model comparison (5 classifiers) |
| **3** | Can we deploy a saved model for inference? | joblib Pipeline + CLI scripts |
| **4** | How do title vs. body text contribute? | Dual TF-IDF vectorizers |

---

## Model performance

Test set: ~33,192 reviews. Best model from notebook evaluation: **SVM** at **93% accuracy**.

| Model | Test accuracy |
|-------|---------------|
| SVM (selected for deployment) | **93%** |
| Random Forest | 92% |
| Logistic Regression | 90% |
| Decision Tree | 91% |
| Naive Bayes | 93% |

Production script (`src/train_model.py`) uses Random Forest on the full dataset; retrain with SVM in the notebook for maximum accuracy.

---

## Repository structure

```
.
├── data/
│   └── product_reviews_full.csv
├── model/
│   └── sentiment_model.pkl
├── notebook/
│   └── product_reviews_analysis.ipynb
├── src/
│   ├── train_model.py
│   └── test_model.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/Marko-Vuchko/product-review-sentiment-analysis.git
cd product-review-sentiment-analysis
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train_model.py
```

### 3. Test inference

```bash
python src/test_model.py
```

### 4. Explore the notebook

```bash
jupyter notebook notebook/product_reviews_analysis.ipynb
```

---

## Methodology

1. **Data loading** — Load review CSV; drop missing values; normalize sentiment labels.
2. **Feature engineering** — TF-IDF on `review_title` and `review_text`; numeric `review_length`.
3. **Preprocessing** — `ColumnTransformer` + `Pipeline` for reproducible train/inference parity.
4. **Model selection** — Compare Logistic Regression, Naive Bayes, Decision Tree, Random Forest, SVM.
5. **Deployment** — Serialize pipeline with joblib; load in `test_model.py` for batch or interactive scoring.

---

## Tech stack

**Python:** pandas, scikit-learn, joblib, Matplotlib, Seaborn, Jupyter

---

## Author

**Marko Vučković** — Data Analyst & Developer  
[GitHub](https://github.com/Marko-Vuchko) · [Email](mailto:markovucko12@gmail.com)

---

## License

This project is released under the [MIT License](LICENSE).
